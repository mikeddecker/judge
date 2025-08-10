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
                        total_batch_loss = head.compute_loss(outputs, batch_y, batch_mask)
                        val_loss += total_batch_loss.item()

                        # Accuracy
                        current_f1 = head.update_metrics(outputs, batch_y, batch_mask)

                metrics = head.compute_metrics()
                head.reset_metrics()

            return {
                'val_loss' : val_loss / len(dataloader),
                'f1_total_avg' : current_f1,
                'metrics' : metrics
            }
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
            
            bestModelStats = load_json_file(bestModelJsonStatsPath)

            df_layers, df_composition, max_instances_per_role = self.repo.get_recognition_config()
            backbone_output_neurons = PYTORCH_MODELS_SKILLS_TEST[modelname].get_output_feature_dim(recipe)
            head = OutputHeadRecognition(backbone_output_neurons, df_layers, df_composition, max_instances_per_role)
            model = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(device)
            optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.2)
            
            classification_reports = {}
            epoch_start = 0
            losses_over_time = []
            metrics_over_time = {}
            modelstats = {}
            f1_avgs_over_time = []
            if not from_scratch and os.path.exists(checkpointPath) and os.path.exists(modelstatsPath):
                checkpoint = torch.load(checkpointPath, weights_only=False)
                modelstats = load_json_file(modelstatsPath)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_start = modelstats['epoch'] + 1
                losses_over_time = modelstats['losses_over_time']
                f1_avgs_over_time = modelstats['f1_total_avgs_over_time']
                metrics_over_time = {} if 'metrics_over_time' not in modelstats.keys() else modelstats['metrics_over_time']
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
            # Adapting the losses_over_time, as limiting to 10% can change occurences of faults, bodyrotations... a little
            # TODO : load previous model instead of current :)
            print("Update re-evaluate and use it to update best models")
            
            # TODO : re-add weighted losses_over_time

            revalidation_results = validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
            validation_loss = revalidation_results['val_loss']
            print(f"Re-eval loss {validation_loss}")
            print(f"Re-eval f1_avg: {revalidation_results['f1_total_avg']:.4f}")

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
                
                validation_results = validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                val_loss = validation_results['val_loss']
                print(f"Epoch {epoch+1}, Train Loss: {total_loss / len(dataloaderTrain):.4f}")
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")
                print(f"Epoch {epoch+1}, Validation f1_avg: {validation_results['f1_total_avg']:.4f}")
                
                # Call the epoch end self, because it is not called by DataLoader, although it shuffles.
                # train_generator.on_epoch_end()

                losses_over_time.append(val_loss)
                f1_avgs_over_time.append(validation_results['f1_total_avg'])
                scheduler.step(val_loss)
                metrics_over_time[str(epoch)] = validation_results['metrics']
                
                minIndexAcc = f1_avgs_over_time.index(max(f1_avgs_over_time))
                hasValAccImproved = len(losses_over_time) - minIndexAcc - 1 == 0
                epochsNoImprovement = len(losses_over_time) - minIndexAcc - 1

                patience = 5
                if epochsNoImprovement > patience:
                    print(f"No improvement for {epochsNoImprovement} epochs - stopping")
                    break

                # TODO : add .current.json & compare to previous run
                if hasValAccImproved:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpointPath)

                    stats = {
                        'epoch': epoch,
                        'validation_results': validation_results,
                        'f1_total_avgs_over_time' : f1_avgs_over_time,
                        'metrics_over_time': metrics_over_time,
                        'losses_over_time': losses_over_time,
                        'classification_reports' : None,
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

                    if validation_results['f1_total_avg'] > revalidation_results['f1_total_avg']:
                        print(f"Model {modelname} improved from {revalidation_results['f1_total_avg']} to {validation_results['f1_total_avg']}")
                        with open(modelstatsPath, "w") as fp:
                            json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                        torch.save(model.state_dict(), path)

                    if not bestModelStats or max(stats['f1_total_avgs_over_time']) > max(bestModelStats['f1_total_avgs_over_time']):
                        if bestModelStats:
                            print(f"Model {modelname} improved the previous best model {bestModelStats['modelname']} from {revalidation_results['f1_total_avg']} to {validation_results['f1_total_avg']}")
                        with open(bestModelJsonStatsPath, "w") as fp:
                            json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                        torch.save(model.state_dict(), pathBest)
            
        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()

def collate_fn_skills(batch):
    batch_X, batch_y, batch_mask = zip(*batch)
    return torch.stack(batch_X), list(batch_y), list(batch_mask)

