#!/usr/bin/env python
# coding: utf-8
import json
import os
import pandas as pd
import sqlalchemy as sqlal
import torch

from collections import defaultdict
from constants import ENVS

from helpers import NumpyTypeEncoder
from managers.Repository import DataRepository
from types import SimpleNamespace

class RepoModels(DataRepository):
    VideoNames = {} # pandas dataframe

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_model(self, recipe: SimpleNamespace):
        """For revalidation"""
        raise NotImplementedError()
    
    def save_model_checkpoint(
            self, 
            recipe: SimpleNamespace, 
            validation_results: dict, 
            model: torch.nn.Module,
            isBestOfRecipe: bool,
            optimizer = None,
            scheduler = None,
            testrun : bool=False,
        ):
        """
        Save model and optimizer/scheduler state dicts.
        
        Parameters:
            recipe (SimpleNamespace): contains information about naming
            validation_results (dict): validation metrics
            model (torch.nn.Module): the model to save
            isBestOfRecipe (bool): Also save as best weights
            optimizer: optimizer state (optional)
            scheduler: learning rate scheduler state (optional)
            testrun (bool): whether this is a test run
        """
        saveRounds = [True, False] if isBestOfRecipe else [True]

        for isCheckpoint in saveRounds:
            filename_parts = [recipe.model]
            if isCheckpoint:
                filename_parts.append('checkpoint')
            if testrun:
                filename_parts.append('testrun')
            filename_partial_text = '.'.join(filename_parts)
            modelpath = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{filename_partial_text}.state_dict.pt")
            resultpath = os.path.join(ENVS.DIRS.WEIGHTS.SKILLS, f"{filename_partial_text}.stats.json")

            state_dict = {
                'model_state_dict': model.state_dict(),
            }

            if optimizer is not None:
                state_dict['optimizer_state_dict'] = optimizer.state_dict()

            if scheduler is not None:
                state_dict['scheduler_state_dict'] = scheduler.state_dict()

            torch.save(state_dict, modelpath)

            with open(resultpath, "w") as fp:
                json.dump(validation_results, fp, indent=4, cls=NumpyTypeEncoder, sort_keys=True)

REPO_MODELS = RepoModels()

