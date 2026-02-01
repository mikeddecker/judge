import functools
import gc
import json
import os
import torch
import torch.optim as optim
import numpy as np
import time
import random

from base_utils import load_json_file
from collections import defaultdict
from colorama import Fore, Style
from datetime import datetime
from domain.types import IsBestOfDict
from constants import RECIPES, SPEEDMODES, ENVS, PYTORCH_MODELS_SKILLS
from dotenv import load_dotenv
from helpers import localize_get_best_modelpath
from localizor_with_strats import predict_and_save_locations
from managers.RepoGeneral import REPO_GENERAL
from managers.RepoModels import REPO_MODELS
from managers.RepoStats import REPO_STATS
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
from models.OutputHeadRecognition import OutputHeadRecognition
from pprint import pprint
from torch.utils.data import DataLoader
from tqdm import tqdm
from types import SimpleNamespace

DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
MAX_EPOCHS = 300
MAX_EPOCHS_TESTRUN = 5
SCHEDULER_PATIENCE = 2
PATIENCE = 5

DEFAULT_COMPARE_METHOD_IS_BEST_MODEL = 'quadratic_validation_length_weighted_f1'
DEFAULT_COMPARE_METHOD_HAS_MODEL_IMPROVED = 'f1_avg'

print(f"Using DEVICE: {DEVICE}")
load_dotenv()

torch.backends.cudnn.benchmark = True

class TrainerSkills:
    _optimizer = None
    _testrun: bool = False
    _step = 'SKILL'

    def __init__(self, testrun:bool=False):
        self._testrun=testrun
        self._max_epochs = MAX_EPOCHS_TESTRUN if self._testrun else MAX_EPOCHS

    def train(self, recipe: SimpleNamespace, job_arguments:dict={}, speedmode=SPEEDMODES[1]):
        modelname = recipe.model
        if modelname not in PYTORCH_MODELS_SKILLS.keys():
            raise ValueError(modelname)
        
        rundate = datetime.now()
        start = time.time()

        try:
            self.__create_or_recreate_cropped_videos(speedmode=speedmode)
            head = OutputHeadRecognition(recipe)
            DefaultGeneratorSkills = functools.partial(
                DataGeneratorSkills, 
                head=head,
                testrun=self._testrun,
                recipe=recipe,
            )
            train_generator = DefaultGeneratorSkills(train_test_val="train")
            val_generator = DefaultGeneratorSkills(train_test_val="val")
            dataloader_train = DataLoader(train_generator, recipe.batch_size, shuffle=True)
            dataloader_val = DataLoader(val_generator, recipe.batch_size, shuffle=True)
            
            revalidate_if_required = speedmode == SPEEDMODES[2]
            if revalidate_if_required:
                self.__revalidate_previous_runs(datetime.now(), dataloader_val)

            model: torch.nn.Module = PYTORCH_MODELS_SKILLS[modelname](head=head, recipe=recipe).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=recipe.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=0.2)

            train_result_id = REPO_STATS.add_train_result(
                recipe=recipe,
                testrun=self._testrun
            )

            current_epoch = 0
            validation_results_best_epoch = None
            # Training loop
            print(f"{Fore.LIGHTMAGENTA_EX}STARTING TO TRAIN FROZEN + UNFROZEN:{Style.RESET_ALL}")
            for frozen_run_pre_trained_weights in [True, False]:
                if not frozen_run_pre_trained_weights:
                    for param in model.parameters():
                        param.requires_grad = True

                for epoch in range(current_epoch + 1, self._max_epochs + 1):
                    current_epoch = epoch
                    print(f"============= EPOCH {current_epoch} =============")
                    self.__train_epoch(model, dataloader_train, optimizer, head)
                    validation_results = self.__validate(model=model, dataloader=dataloader_val)
                    validation_results = { **validation_results, 'length_train': len(dataloader_train) }
                    scheduler.step(validation_results['val_loss'])
                    
                    REPO_STATS.save_epoch_results(train_result_id, epoch, validation_results)

                    if current_epoch == 1:
                        epochsNoImprovement = 0
                        validation_results_best_epoch = validation_results
                    else:
                        model_in_training_improved = REPO_STATS.compare_validation_results(
                            current_results=validation_results,
                            other_results=validation_results_best_epoch,
                            method=job_arguments.get('has_epoch_improved_method', DEFAULT_COMPARE_METHOD_HAS_MODEL_IMPROVED)
                        )
                        if model_in_training_improved:
                            validation_results_best_epoch = validation_results
                            epochsNoImprovement = 0
                            REPO_STATS.update_train_result(
                                train_result_id=train_result_id,
                                updated_params={ 'bestEpoch': epoch }
                            )
                        else:
                            epochsNoImprovement += 1

                    model_is_best_of : IsBestOfDict = REPO_STATS.check_is_best_model(
                        train_result_id=train_result_id,
                        recipe=recipe,
                        validation_results=validation_results,
                        step=self._step,
                        compare_result_method=job_arguments.get('is_best_model_method', DEFAULT_COMPARE_METHOD_IS_BEST_MODEL),
                        isTestrun=self._testrun,
                    )

                    pprint(f'Using {job_arguments.get('is_best_model_method', DEFAULT_COMPARE_METHOD_IS_BEST_MODEL)}:  model is best of')
                    pprint(model_is_best_of)

                    REPO_MODELS.save_model_checkpoint(
                        recipe=recipe,
                        model=model,
                        validation_results=validation_results,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        testrun=self._testrun,
                        isBestOfRecipe=model_is_best_of['recipe']
                    )

                    print(f"🛈 Epochs no improvement: {epochsNoImprovement} (Patience: {PATIENCE})")
                    if epochsNoImprovement > PATIENCE:
                        print(f"💬✋ Stopping - No improvement for {PATIENCE} epochs - (Frozen run = {frozen_run_pre_trained_weights})")
                        break

            # End of train loop
            REPO_STATS.update_train_result(
                train_result_id=train_result_id,
                updated_params={ 'trainEnd' : datetime.now() }
            )

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    def __create_or_recreate_cropped_videos(self, speedmode: str):
        unique_videoIds = REPO_GENERAL.get_videoIds_of_videos_with_skills()
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
            videoIds=new_videos.union(existing_redo_subset),
            saveAsVideo=True
        )

    def __validate(self, model, dataloader) -> dict:
        """
        Validates the current model and adds val_loss to the epoch metrics

        Returns: {
            "metric_per_layer": {'acc': {'prop1': list[float]|float, ...}, 'f1': {...}'},
            "metric_avg_of_layers": {'acc': float, 'f1': float, ...},
            "confusion_heads": self.confusion_heads,
            'val_loss': float,
            'val_length': int,
        }
        """
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
            **metrics,
            'val_loss' : val_loss / len(dataloader),
            'val_length': len(dataloader.dataset),
        }

    def __train_epoch(self, model, dataloader, optimizer, head: OutputHeadRecognition):
        model.train()
        total_loss = 0.0
        i = 0
        for batch_X, batch_y, batch_mask, skill_id in tqdm(dataloader):
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

    def __revalidate_previous_runs(self, rundate, dataloader_val):
        raise NotImplementedError('__revalidate_previous_runs')

