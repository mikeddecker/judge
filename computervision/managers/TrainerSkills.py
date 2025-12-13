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

from collections import defaultdict
from colorama import Fore, Style
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
        elif hasattr(obj, "ndim"):  # torch.Tensor
            return obj.item() if obj.ndim == 0 else obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, SimpleNamespace):
            return obj.__dict__
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return super().default(obj)

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler()

class TrainerSkills:
    def __init__(self):
        self.repo = DataRepository()

    def train(self, recipe: SimpleNamespace, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, patience:int=5, speedmode=SPEEDMODES[1]):
        rundate = date.today().strftime('%Y%m%d')

        #########################################################################################
        def validate(model, dataloader, optimizer, device='cuda'):
            model.eval()
            val_loss = 0.0

            head : OutputHeadRecognition = model.head

            with torch.no_grad():
                for batch_X, batch_y, batch_mask, skill_id in tqdm(dataloader):
                    with torch.amp.autocast(device_type=device):
                        optimizer.zero_grad()
                        outputs = model(batch_X / 255)

                        try:
                            
                            # Loss
                            total_batch_loss = head.compute_loss(outputs, batch_y, batch_mask)
                            val_loss += total_batch_loss.item()

                            # Accuracy
                            current_f1 = head.update_metrics(outputs, batch_y, batch_mask)
                        except Exception as e:
                            print(f"❌ Error during validation on skill ☣️ {skill_id} ☣️")
                            raise e

                metrics = head.compute_metrics()
                head.reset_metrics()

            return {
                'val_loss' : val_loss / len(dataloader),
                'f1_total_avg' : float(np.mean(list(metrics['f1'].values()))),
                'accuracy_avg' : float(np.mean(list(metrics['acc'].values()))),
                'precision_avg' : float(np.mean(list(metrics['precision'].values()))),
                'recall_avg' : float(np.mean(list(metrics['recall'].values()))),
                'metrics' : metrics,
                'confusion_values' : head.confusion_values,
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
            scheduler_patience = 2
            start = time.time()
            testrun = False
            modelname = recipe.model
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)

            create_or_recreate_cropped_videos(speedmode=speedmode)

            os.makedirs(os.path.join(ENVS.DIRS.WEIGHTS.SKILLS), exist_ok=True)
            path = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}.state_dict.pt")
            pathBest = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"best.state_dict.pt")
            checkpointPath = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.checkpoint.pt")
            modelstatsPath = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.stats.json")
            modelstatsPathCurrent = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.stats.current.json")
            best_stats_path = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"best{'_testrun' if testrun else ''}.stats.json")
            
            df_layers, df_composition, max_instances_per_role = self.repo.get_recognition_config()
            backbone_output_neurons = PYTORCH_MODELS_SKILLS_TEST[modelname].get_output_feature_dim(recipe)
            prop_counts = self.repo.get_skill_prop_counts()
            head = OutputHeadRecognition(backbone_output_neurons, df_layers, df_composition, max_instances_per_role, prop_counts)
            
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
            
            try:
                best_model_revalidation_results = None
                best_model_name = None
                revalidation_results = None
                if not from_scratch and os.path.exists(pathBest):
                    best_model_stats = load_json_file(best_stats_path)
                    best_model_name = best_model_stats['modelname']
                    best_model_backbone_output_neurons = PYTORCH_MODELS_SKILLS_TEST[best_model_name].get_output_feature_dim(recipe)
                    best_head = OutputHeadRecognition(best_model_backbone_output_neurons, df_layers, df_composition, max_instances_per_role, prop_counts)
                    try:                        
                        model: torch.nn.Module = PYTORCH_MODELS_SKILLS[best_model_name](head=best_head, recipe=RECIPES['SKILL'][best_model_stats['recipe']['name']]).to(device)
                        model.load_state_dict(torch.load(pathBest, weights_only=True))
                        optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=0.2)
                        print(f"Re-evaluate best of best model ({best_model_name}), to get the most optimal comparisons")
                        best_model_revalidation_results = validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                        print(f"{Fore.YELLOW}Target best ({best_model_name}) - {best_model_revalidation_results['f1_total_avg']:.4f}{Style.RESET_ALL}")
                    except:
                        print(f"{Fore.RED}Error loading (weigths of) best model.{Style.RESET_ALL} Highly likely because of new parameters, this run becomes default best")

                if not from_scratch and os.path.exists(path):
                    model: torch.nn.Module = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(device)
                    model.load_state_dict(torch.load(path, weights_only=True))
                    optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=0.2)
                    print(f"Re-evaluate best of current model {modelname}, to get the most optimal comparisons")
                    revalidation_results = validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                    print(f"{Fore.MAGENTA}Target {modelname}: {revalidation_results['f1_total_avg']:.4f}{Style.RESET_ALL}")
            except RuntimeError as e:
                if "size mismatch" not in str(e):
                    raise e
            except Exception as e:
                print("revalidation went wrong")
                raise e
            
            # For new training rounds
            model: torch.nn.Module = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(device)
            optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=scheduler_patience, factor=0.2)

            classification_reports = {}
            epoch_start = 0
            losses_over_time = []
            metrics_over_time = {}
            modelstats = {}
            f1_avgs_over_time = []
            acc_avgs_over_time = []
            if not from_scratch and os.path.exists(checkpointPath) and os.path.exists(modelstatsPathCurrent):
                checkpoint = torch.load(checkpointPath, weights_only=False)
                modelstats = load_json_file(modelstatsPathCurrent)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_start = modelstats['epoch'] + 1
                losses_over_time = modelstats['losses_over_time']
                f1_avgs_over_time = modelstats['f1_total_avgs_over_time']
                acc_avgs_over_time = modelstats['acc_avgs_over_time']
                metrics_over_time = {} if 'metrics_over_time' not in modelstats.keys() else modelstats['metrics_over_time']
                classification_reports = {} if 'classification_reports' not in modelstats.keys() else modelstats['classification_reports']

            if unfreeze_all_layers:
                for param in model.parameters():
                    param.requires_grad = True

            # Training loop
            for epoch in range(epoch_start, epochs + epoch_start):
                print(f"============= EPOCH {epoch} =============")

                model.train()
                total_loss = 0.0
                i = 0
                for batch_X, batch_y, batch_mask, skill_id in tqdm(dataloaderTrain):
                    with torch.amp.autocast(device_type='cuda'):
                        optimizer.zero_grad()  # Clear gradients
                        
                        # Forward pass
                        outputs = model(batch_X / 255)
                        total_batch_loss = head.compute_loss(outputs, batch_y, batch_mask, skillId=skill_id)
                        if total_batch_loss.requires_grad:
                            total_batch_loss.backward()
                            optimizer.step()
                        else:
                            # Allow to continue training, but display a warning
                            print(f"⚠️ Warning (Skill {skill_id}): loss tensor has no grad, skipping batch")
                            pprint({ k: v for k, v in batch_y.items() if k.startswith("composition_") })
                            pprint({ k: v for k, v in batch_mask.items() if k.startswith("composition_") })
                            continue
                    
                    total_loss += total_batch_loss.item()
                    i+=1
                
                validation_results = validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                val_loss = validation_results['val_loss']
                
                # Call the epoch end self, because it is not called by DataLoader, although it shuffles.
                # train_generator.on_epoch_end()

                losses_over_time.append(val_loss)
                f1_avgs_over_time.append(validation_results['f1_total_avg'])
                acc_avgs_over_time.append(validation_results['accuracy_avg'])
                scheduler.step(val_loss)
                metrics_over_time[str(epoch)] = validation_results['metrics']
                
                minIndexF1 = f1_avgs_over_time.index(max(f1_avgs_over_time))
                minIndexLoss = losses_over_time.index(min(losses_over_time))
                hasValF1Improved = len(losses_over_time) - minIndexF1 - 1 == 0
                hasValLossImproved = len(losses_over_time) - minIndexLoss - 1 == 0
                epochsNoImprovement = len(losses_over_time) - max(minIndexF1, minIndexLoss) - 1

                color_acc = Fore.GREEN if hasValF1Improved else Fore.RED
                color_loss = Fore.GREEN if hasValLossImproved else Fore.RED
                print(f"Epoch {epoch}, Train Loss: {total_loss / len(dataloaderTrain):.4f}")
                print(f"Epoch {epoch}, Validation Loss: {color_loss}{val_loss:.4f}{Style.RESET_ALL}")
                print(f"Epoch {epoch}, Validation f1_avg: {color_acc}{validation_results['f1_total_avg']:.4f}{Style.RESET_ALL}")

                if epochsNoImprovement > patience:
                    print(f"No improvement for {epochsNoImprovement} epochs - stopping")
                    break

                # TODO : add .current.json & compare to previous run
                if hasValF1Improved or hasValLossImproved:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                    }, checkpointPath)

                    stats = {
                        'epoch': epoch,
                        'validation_results': validation_results,
                        'f1_total_avgs_over_time' : f1_avgs_over_time,
                        'acc_avgs_over_time' : acc_avgs_over_time,
                        'metrics_over_time': metrics_over_time,
                        'losses_over_time': losses_over_time,
                        'classification_reports' : None,
                        'confusion_matrix': None,
                        'final_classification_reports' : None,
                        'time' : time.time() - start,
                        'length_train': len(train_generator.Skills),
                        'length_val': len(val_generator.Skills),
                        'rundate': rundate,
                        'modelname': modelname,
                        'recipe': recipe
                    }
                    
                    with open(modelstatsPathCurrent, "w") as fp:
                        json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                    if not from_scratch and (not revalidation_results or validation_results['f1_total_avg'] > revalidation_results['f1_total_avg']):
                        if revalidation_results:
                            print(f"Current model {modelname} improved from {revalidation_results['f1_total_avg']} to {validation_results['f1_total_avg']}")
                        with open(modelstatsPath, "w") as fp:
                            json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)
                        
                        torch.save(model.state_dict(), path)

                    if not from_scratch and (not best_model_revalidation_results or validation_results['f1_total_avg'] > best_model_revalidation_results['f1_total_avg']):
                        if best_model_revalidation_results:
                            print(f"Model {modelname} improved the previous best model {best_model_name} from {best_model_revalidation_results['f1_total_avg']} to {validation_results['f1_total_avg']}")
                        with open(best_stats_path, "w") as fp:
                            json.dump(stats, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

                        torch.save(model.state_dict(), pathBest)
            
        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()

