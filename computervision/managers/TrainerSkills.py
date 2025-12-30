import functools
import gc
import json
import os
import torch
import torch.optim as optim
import numpy as np
import time
import random

from collections import defaultdict
from colorama import Fore, Style
from constants import RECIPES, SPEEDMODES, ENVS, PYTORCH_MODELS_SKILLS
from dotenv import load_dotenv
from helpers import NumpyTypeEncoder
from managers.DataRepository import REPO
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
from pprint import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import date
from base_utils import load_json_file
from models.OutputHeadRecognition import OutputHeadRecognition
from types import SimpleNamespace
from localizor_with_strats import predict_and_save_locations
from helpers import localize_get_best_modelpath

DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
SCHEDULER_PATIENCE = 1

print(f"Using DEVICE: {DEVICE}")
load_dotenv()

torch.backends.cudnn.benchmark = True

class TrainerSkills:
    _optimizer = None

    def train(self, recipe: SimpleNamespace, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, patience:int=3, speedmode=SPEEDMODES[1]):
        rundate = date.today().strftime('%Y%m%d')
        start = time.time()

        try:
            testrun = False
            epochs = 5 if testrun else epochs
            modelname = recipe.model
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)

            self.__create_or_recreate_cropped_videos(speedmode=speedmode)

            os.makedirs(os.path.join(ENVS.DIRS.WEIGHTS.SKILLS), exist_ok=True)
            path = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.state_dict.pt")
            pathBest = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"best{'_testrun' if testrun else ''}.state_dict.pt")
            checkpointPath = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.checkpoint.pt")
            modelstatsPath = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.stats.json")
            modelstatsPathCurrent = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{modelname}{'_testrun' if testrun else ''}.stats.current.json")
            best_stats_path = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"best{'_testrun' if testrun else ''}.stats.json")
            
            df_layers, df_composition, max_instances_per_role = REPO.get_recognition_config()
            backbone_output_neurons = PYTORCH_MODELS_SKILLS[modelname].get_output_feature_dim(recipe)
            prop_counts = REPO.get_skill_prop_counts()
            head = OutputHeadRecognition(backbone_output_neurons, df_layers, df_composition, max_instances_per_role, prop_counts)
            
            DefaultGeneratorSkills = functools.partial(
                DataGeneratorSkills, 
                frameloader=FrameLoader(REPO),
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
                    best_model_backbone_output_neurons = PYTORCH_MODELS_SKILLS[best_model_name].get_output_feature_dim(recipe)
                    best_head = OutputHeadRecognition(best_model_backbone_output_neurons, df_layers, df_composition, max_instances_per_role, prop_counts)
                    try:                        
                        model: torch.nn.Module = PYTORCH_MODELS_SKILLS[best_model_name](head=best_head, recipe=RECIPES['SKILL'][best_model_stats['recipe']['name']]).to(DEVICE)
                        model.load_state_dict(torch.load(pathBest, weights_only=True))
                        optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
                        print(f"Re-evaluate best of best model ({best_model_name}), to get the most optimal comparisons")
                        best_model_revalidation_results = self.__validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                        print(f"{Fore.YELLOW}Target best ({best_model_name}) - {best_model_revalidation_results['f1_total_avg']:.4f}{Style.RESET_ALL}")
                    except:
                        print(f"{Fore.RED}Error loading (weigths of) best model.{Style.RESET_ALL} Highly likely because of new parameters, this run becomes default best")

                if not from_scratch and os.path.exists(path):
                    model: torch.nn.Module = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(DEVICE)
                    model.load_state_dict(torch.load(path, weights_only=True))
                    optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=0.2)
                    print(f"Re-evaluate best of current model {modelname}, to get the most optimal comparisons")
                    revalidation_results = self.__validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
                    print(f"{Fore.MAGENTA}Target {modelname}: {revalidation_results['f1_total_avg']:.4f}{Style.RESET_ALL}")
            except RuntimeError as e:
                if "size mismatch" not in str(e) and "Missing key(s) in state_dict" not in str(e):
                    raise e
            except Exception as e:
                print("revalidation went wrong")
                raise e
            
            # For new training rounds
            model: torch.nn.Module = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=0.2)

            epoch_start = 0
            losses_over_time = []
            metrics_over_time = {}
            modelstats = {}
            f1_avgs_over_time = []
            f1_avg_over_losses_over_time = []
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
                f1_avg_over_losses_over_time = [] if 'f1_avg_over_losses_over_time' not in modelstats.keys() else modelstats['f1_avg_over_losses_over_time']
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
                    with torch.amp.autocast(device_type=DEVICE_TYPE):
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
                
                validation_results = self.__validate(model=model, dataloader=dataloaderVal, optimizer=optimizer)
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

                # Testing f1_avg over loss improvement for early stopping
                f1_avg_over_loss = validation_results['f1_total_avg'] / val_loss
                f1_avg_over_losses_over_time.append(f1_avg_over_loss)
                hasAccOverLossImproved = len(f1_avg_over_losses_over_time) - 1 == f1_avg_over_losses_over_time.index(max(f1_avg_over_losses_over_time))
                f1_avg_over_loss_improvement = ((f1_avg_over_loss / f1_avg_over_losses_over_time[-2]) if len(f1_avg_over_losses_over_time) > 1 else 0) - 1

                color_acc = Fore.GREEN if hasValF1Improved else Fore.RED
                color_loss = Fore.GREEN if hasValLossImproved else Fore.RED
                color_f1_avg_over_loss = Fore.GREEN if hasAccOverLossImproved else Fore.RED
                print(f"Epoch {epoch}, Train Loss: {total_loss / len(dataloaderTrain):.4f}")
                print(f"Epoch {epoch}, Validation Loss: {color_loss}{val_loss:.4f}{Style.RESET_ALL}")
                print(f"Epoch {epoch}, Validation f1_avg: {color_acc}{validation_results['f1_total_avg']:.4f}{Style.RESET_ALL}")
                print(f"Epoch {epoch}, Validation f1_avg / loss: {color_f1_avg_over_loss}{f1_avg_over_loss:.4f}{Style.RESET_ALL}")
                if len(f1_avg_over_losses_over_time) > 1:
                    print(f"Epoch {epoch}, f1_avg / loss improvement: {f1_avg_over_loss_improvement:.2%}%")

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
                        'f1_avg_over_losses_over_time' : f1_avg_over_losses_over_time,
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

    def __validate(self, model, dataloader, optimizer):
        model.eval()
        val_loss = 0.0

        head : OutputHeadRecognition = model.head

        with torch.no_grad():
            for batch_X, batch_y, batch_mask, skill_id in tqdm(dataloader):
                with torch.amp.autocast(device_type=DEVICE_TYPE):
                    outputs = model(batch_X / 255)
                    try:
                        val_loss += head.compute_loss(outputs, batch_y, batch_mask).item() # Loss
                        head.update_metrics(outputs, batch_y, batch_mask) # Accuracy
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
            'confusion_heads' : head.confusion_heads,
        }

    def __create_or_recreate_cropped_videos(self, speedmode: str):
        unique_videoIds = REPO.get_videoIds_of_videos_with_skills()
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
            repo=REPO,
            videoIds=new_videos.union(existing_redo_subset),
            saveAsVideo=True
        )

