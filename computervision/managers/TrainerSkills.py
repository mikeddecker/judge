import functools
import gc
import json
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import random

from constants import ENVS, PYTORCH_MODELS_SKILLS, PYTORCH_MODELS_SKILLS_TEST
from dotenv import load_dotenv
from managers.DataRepository import DataRepository
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from datetime import datetime, date
from helpers import weighted_mse_loss
from base_utils import load_json_file, dump_json_file
from models.OutputHeadRecognition import OutputHeadRecognition
from types import SimpleNamespace
from localizor_with_strats import predict_and_save_locations
from helpers import localize_get_best_modelpath

from constants import RECIPES, SPEEDMODES

class NumpyTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler()

class TrainerSkills:
    def __init__(self):
        self.repo = DataRepository()

    def __compute_losses(self, outputs, batch_y, loss_fns):
        """Outputs is a ModelDict, acts like a dictionary"""
        losses = []

        for key, pred in outputs.items():
            target = batch_y[key]
            
            if key in ['Skill', 'Turner1', 'Turner2', 'Type']:  # Categorical
                loss = loss_fns[key](pred, target.long())
            else:
                loss = loss_fns[key](pred.squeeze(), target)
            
            losses.append(loss)
        return sum(losses)

    def validate_old(self, model, dataloader, optimizer, loss_fns, target_names, device='cuda'):
        model.eval()
        val_loss = 0.0

        skillconfig: dict = {}
        y_pred = { key : [] for key, _ in skillconfig.items() }
        y_true = { key : [] for key, _ in skillconfig.items() }

        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader):
                with torch.amp.autocast(device_type=device):
                    optimizer.zero_grad()
                    outputs = model(batch_X / 255)
                    # Loss
                    total_batch_loss = self.__compute_losses(outputs=outputs, batch_y=batch_y, loss_fns=loss_fns)
                    val_loss += total_batch_loss.item()

                    # Accuracy
                    for key, pred in outputs.items():
                        target = batch_y[key]

                        valueType = skillconfig[key][0]
                        if valueType == "Categorical":
                            pred = F.softmax(pred, dim=1)
                            _, pred = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                        elif valueType == "Numerical":
                            maxValue = skillconfig[key][2]
                            pred = torch.round(pred * maxValue).squeeze(dim=0).type(torch.int64)
                            target = torch.round(target * maxValue).type(torch.int64)
                        else:
                            pred = torch.round(pred).squeeze(dim=0).type(torch.int64)
                            target = torch.round(target).type(torch.int64)
                        
                        y_pred[key].extend(pred.data.cpu().numpy())
                        y_true[key].extend(target.data.cpu().numpy())
                    
        print(f"="*80)
        classification_reports = {}
        for key in y_true.keys():
            classKey = key if key not in ['Turner1', 'Turner2'] else 'Turner'
            labels = None if classKey not in target_names.keys() else range(1, len(target_names[classKey]) + 1) # Mysql startIdx = 1
            tn = None if classKey not in target_names.keys() else target_names[classKey]
            classification_reports_string = classification_report(y_true[key], y_pred[key], labels=labels, target_names=tn, zero_division=0)
            classification_reports[key] = classification_report(y_true[key], y_pred[key], output_dict=True, labels=labels, target_names=tn, zero_division=0)
            print(f"----- Details {key} ----")
            print(classification_reports_string)

            lbls = labels if labels is not None else range(max(max(y_true[key]), max(y_pred[key])) + 1)
            cm = confusion_matrix(y_true[key], y_pred[key], labels=lbls)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            print("Confusion Matrix:", key)
            print(cm_df)
            print(f"="*80)

        f1_scores_epoch = { k: class_report['macro avg']['f1-score'] for k, class_report in classification_reports.items() }
        f1_scores_epoch["Total"] = sum(f1_scores_epoch.values()) / len(f1_scores_epoch)

        print(f"Total skill (macro avg) accuracy", classification_reports['Skill']['macro avg'])
        print(f"Total f1 score", sum(f1_scores_epoch.values()) / len(f1_scores_epoch))
        return val_loss / len(dataloader), f1_scores_epoch, classification_reports, cm

    def train(self, recipe: SimpleNamespace, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, speedmode=SPEEDMODES[1]):
        rundate = date.today().strftime('%Y%m%d')

        #########################################################################################
        def validate(model, dataloader, optimizer, device='cuda'):
            model.eval()
            val_loss = 0.0

            head : OutputHeadRecognition = model.head

            with torch.no_grad():
                for batch_X, batch_y, batch_mask in tqdm(dataloader):
                    with torch.amp.autocast(device_type=device):
                        optimizer.zero_grad()
                        outputs = model(batch_X / 255)

                        # Loss
                        total_batch_loss = OutputHeadRecognition.compute_loss(outputs, batch_y, batch_mask)
                        val_loss += total_batch_loss.item()

                        # Accuracy
                        # TODO
                        accaracies = { 'Total': 0.0 }
            return val_loss / len(dataloader), accaracies

        # End validate
        #########################################################################################
        def create_or_recreate_cropped_videos(speedmode):
            unique_videoIds = self.repo.get_videoIds_of_videos_with_skills()
            existing_cropped_videoIds = []
            existing_redo_subset = set()
            new_videos = set()

            for videoId in unique_videoIds:
                vpath = os.path.join(ENVS.DIRS.GENERATED_VIDEODATA, f'{videoId}', f'{videoId}_cropped.mp4')
                if not os.path.exists(vpath):
                    new_videos.add(videoId)
                else:
                    existing_cropped_videoIds.append(videoId)

            recipename, weightpath = localize_get_best_modelpath()
            random.shuffle(existing_cropped_videoIds)
            for i in range(int(np.sqrt(len(existing_cropped_videoIds))) if speedmode == SPEEDMODES[1] else 0):
                existing_redo_subset.add(existing_cropped_videoIds[i])

            print(f"Speedmode={speedmode}: Predict and create videocrops")
            predict_and_save_locations(
                recipename=recipename,
                weights=weightpath,
                repo=self.repo,
                videoIds=new_videos.union(existing_redo_subset),
                saveAsVideo=True
            )
            
        # End video creation
        #########################################################################################
        try:
            start = time.time()
            testrun = False
            modelname = recipe.model
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)

            create_or_recreate_cropped_videos(speedmode=speedmode)

            step = 'SKILL'
            os.makedirs(os.path.join(ENVS.DIRS.WEIGHTS, step.lower()), exist_ok=True)
            path = os.path.join(ENVS.DIRS.WEIGHTS, step.lower(), f"{modelname}.state_dict.pt")
            pathBest = os.path.join(ENVS.DIRS.WEIGHTS, step.lower(), f"best.state_dict.pt")
            checkpointPath = os.path.join(ENVS.DIRS.WEIGHTS, step.lower(), f"{modelname}{'_testrun' if testrun else ''}.checkpoint.pt")
            modelstatsPath = os.path.join(ENVS.DIRS.WEIGHTS, step.lower(), f"{modelname}{'_testrun' if testrun else ''}.stats.json")
            modelstatsPathCurrent = os.path.join(ENVS.DIRS.WEIGHTS, step.lower(), f"{modelname}{'_testrun' if testrun else ''}.stats.current.json")
            bestModelJsonStatsPath = os.path.join(ENVS.DIRS.WEIGHTS, step.lower(), f"best{'_testrun' if testrun else ''}.stats.json")

            bestModelStats = { 'f1_macro_avg_accuracy': 0 }
            
            bestModelStats = load_json_file(bestModelJsonStatsPath)

            df_layers, df_composition, max_instances_per_role = self.repo.get_recognition_config()
            backbone_output_neurons = PYTORCH_MODELS_SKILLS_TEST[modelname].get_output_feature_dim(recipe)
            head = OutputHeadRecognition(backbone_output_neurons, df_layers, df_composition, max_instances_per_role)
            model = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(device)
            optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)
            
            classification_reports = {}
            epoch_start = 0
            f1_scores = {}
            losses = []
            modelstats = {}
            total_accuracies = []
            if not from_scratch and os.path.exists(checkpointPath) and os.path.exists(modelstatsPath):
                checkpoint = torch.load(checkpointPath, weights_only=False)
                modelstats = load_json_file(modelstatsPath)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_start = modelstats['epoch'] + 1
                losses = modelstats['losses']
                total_accuracies = modelstats['total_accuracies']
                f1_scores = {} if 'f1_scores' not in modelstats.keys() else modelstats['f1_scores']
                classification_reports = {} if 'classification_reports' not in modelstats.keys() else modelstats['classification_reports']

            if unfreeze_all_layers:
                for param in model.parameters():
                    param.requires_grad = True

            DefaultGeneratorSkills = functools.partial(
                DataGeneratorSkills, 
                frameloader=FrameLoader(self.repo),
                head=head,
                train_test_val="train",
                dim=(recipe.dim,recipe.dim),
                timesteps=recipe.timesteps,
                testrun=testrun
            )
            train_generator = DefaultGeneratorSkills(train_test_val="train")
            val_generator = DefaultGeneratorSkills(train_test_val="val")
        
            dataloaderTrain = DataLoader(train_generator, recipe.batch_size, shuffle=True)
            dataloaderVal = DataLoader(val_generator, recipe.batch_size, shuffle=True)

            # Re-evaluate to know whether the current run is better than the previous runs
            # Adapting the losses, as limiting to 10% can change occurences of faults, bodyrotations... a little
            # TODO : load previous model instead of current :)
            # TODO : Re-evaluate previous run (because accuracy can have changed by new skills)
            print("Update re-evaluate and use it to update best models")
            # TODO : re-add weighted losses

            validation_loss, f1_macro_avg_scores_epoch_reval = validate(
                model=model,
                dataloader=dataloaderVal,
                optimizer=optimizer
            )

            print(f"Re eval loss {validation_loss}")

            # Training loop
            for epoch in range(epoch_start, epochs + epoch_start):
                print(f"============= EPOCH {epoch} =============")

                model.train()
                total_loss = 0.0
                i = 0
                for batch_X, batch_y, batch_mask in tqdm(dataloaderTrain):
                    with torch.amp.autocast(device_type='cuda'):
                        optimizer.zero_grad()  # Clear gradients
                        
                        # Forward pass
                        outputs = model(batch_X / 255)
                        total_batch_loss = head.compute_loss(outputs, batch_y, batch_mask)
                        total_batch_loss.backward()
                        optimizer.step()
                    
                    total_loss += total_batch_loss.item()
                    i+=1
                
                val_loss, f1_scores_epoch = validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(dataloaderTrain):.4f}")
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
                
                # Call the epoch end self, because it is not called by DataLoader, although it shuffles.
                # train_generator.on_epoch_end()

                losses.append(val_loss)
                total_accuracies.append(f1_scores_epoch['Total'])
                scheduler.step(val_loss)
                f1_scores[f'{epoch}'] = f1_scores_epoch
                
                minIndexLoss = losses.index(min(losses))
                minIndexAcc = total_accuracies.index(max(total_accuracies))
                minIndex = max(minIndexAcc, minIndexLoss)
                epochsNoImprovement = len(losses) - minIndex - 1
                hasValLossImproved = len(losses) - minIndexLoss - 1 == 0
                hasValAccImproved = len(losses) - minIndexAcc - 1 == 0

                patience = 5
                if epochsNoImprovement > patience:
                    print(f"No improvement for {epochsNoImprovement} - stopping")
                    break

                # TODO : add .current.json & compare to previous run
                if hasValLossImproved or hasValAccImproved:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpointPath)

                    stats = {
                        'epoch': epoch,
                        'best_epoch' : epoch,
                        'f1_macro_avg_accuracy' : f1_scores[f'{epoch}']['Total'],
                        'total_accuracies' : total_accuracies,
                        'losses': losses,
                        'f1_scores': f1_scores,
                        'classification_reports' : classification_reports,
                        'confusion_matrix': None,
                        'final_classification_reports' : None,
                        'time' : time.time() - start,
                        'length_train': len(train_generator),
                        'length_val': len(val_generator),
                        'rundate': rundate,
                        'modelname': modelname,
                    }
                    
                    with open(modelstatsPathCurrent, "w") as fp:
                        json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                    if stats['f1_macro_avg_accuracy'] > f1_macro_avg_scores_epoch_reval['Total']:
                        with open(modelstatsPath, "w") as fp:
                            json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                        torch.save(model.state_dict(), path)

                    if not bestModelStats or stats['f1_macro_avg_accuracy'] > bestModelStats['f1_macro_avg_accuracy']:
                        with open(bestModelJsonStatsPath, "w") as fp:
                            json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                        torch.save(model.state_dict(), pathBest)
            
            pprint(f"Current f1 macro avg accuracy: {f1_scores_epoch['Total']}")

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()

def collate_fn_skills(batch):
    batch_X, batch_y, batch_mask = zip(*batch)
    return torch.stack(batch_X), list(batch_y), list(batch_mask)

