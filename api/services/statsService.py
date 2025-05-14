import os
import torch
import yaml
import pandas as pd
from .videoService import VideoService
from domain.folder import Folder
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from helpers.ValueHelper import ValueHelper
from typing import List


class StatsService:
    PROPERTIES = [
        "FolderRepo",
        "VideoRepo",
        "StorageFolder",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        self.FolderRepo = FolderRepository(db=db)
        self.VideoRepo = VideoRepository(db=db)

        if not os.path.exists(storage_folder):
            raise NotADirectoryError(f"StorageFolder {storage_folder} does not exist")
        self.StorageFolder = storage_folder
        
    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name in self.PROPERTIES:
                raise AttributeError(f"Cannot modify {name} once it's set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def getRecognitionResults(self, selectedModel: str):
        results = {}

        # Selected model
        results['selected-model'] = selectedModel
        results['train-time'] = 6852.3

        checkpoint = torch.load(os.path.join('..', 'computervision', 'weights', f"{selectedModel}.checkpoint.pt"), weights_only=False)
        results['f1-scores-val'] = checkpoint["f1_scores"]
        results['f1-scores-test'] = {
            "Type" : 0.51,
            "Rotations" : 0.5,
            "Turner1" : 0.5,
            "Turner2" : 0.51,
            "Skill" : 0.55,
            "Hands" : 0.5,
            "Feet" : 0.5,
            "Turntable" : 0.5,
            "BodyRotations" : 0.5,
            "Backwards" : 0.5,
            "Sloppy" : 0.5,
            "Hard2see" : 0.5,
            "Fault" : 0.5,
            "Total": 0.505,
        }
        
        f1_last_epoch = results['f1-scores-val'][len(results['f1-scores-val']) - 1]
        # All models
        results['modelcomparison'] = {
            'HAR_MViT' : {
                'accuracy': f1_last_epoch["Total"],
                'acc-skills': f1_last_epoch["Skill"],
                'last-trained' : '2025-04-28'
            },
            'HAR_MViT_extra_dense' : {
                'accuracy': 0.133,
                'acc-skills': 0.3112,
                'last-trained' : '2025-04-28'
            }
        }
        results['distributions'] = {
            'skills' : {
                'push-up': {
                    'train': 432,
                    'test': 0,
                    'val': 12
                },
                'frog': {
                    'train': 325,
                    'test': 0,
                    'val': 13
                },
                'jump': {
                    'train': 1586,
                    'test': 0,
                    'val': 132
                }
            }
        }
        

        # Best model
        results['best-model'] = selectedModel        


        return results

    def getLocalizeResults(self, selectedModel: str):
        results = {
            'general' : {},
            'best' : selectedModel,
        }

        basepath = os.path.join('..', 'runs', 'detect')
        results['models'] = os.listdir(basepath)

        for traindir in os.listdir(basepath):
            results[traindir] = {}
            trainpath = os.path.join(basepath, traindir)
            results[traindir]['content'] = os.listdir(trainpath)
            argpath = os.path.join(trainpath, 'args.yaml')
            print(os.path.exists(argpath), argpath)
            if os.path.exists(argpath):
                with open(argpath, 'r') as file:
                    results[traindir]['args'] = yaml.safe_load(file)

            # Process results
            if 'weights' not in results[traindir]['content']:
                continue

            if 'results.csv' not in results[traindir]['content']:
                continue

            print(results[traindir]['args']['model'])
            print(results[traindir]['args']['patience'])
            df_results = pd.read_csv(os.path.join(trainpath, 'results.csv'))
            print(len(df_results))


        # checkpoint = torch.load(os.path.join('..', 'computervision', 'weights', f"{selectedModel}.checkpoint.pt"), weights_only=False)


        return results
