import os
import torch
import glob
import json
import yaml
import pandas as pd
from .videoService import VideoService
from domain.folder import Folder
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from helpers.ValueHelper import ValueHelper
from typing import List
from helpers.ConfigHelper import recognition_get_modelpaths, PYTORCH_MODELS_SKILLS


class StatsService:
    PROPERTIES = [
        "FolderRepo",
        "VideoRepo",
        "StorageFolder",
        "videoService",
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
        results = {
            'best' : {
                'f1-macro-avg' : 0
            },
            'modelcomparison' : {}
        }
      
        results['trainrounds'] = recognition_get_modelpaths()

        for modelname in PYTORCH_MODELS_SKILLS.keys():
            results[modelname] = {
                'best' : {
                    'f1-macro-avg' : 0
                }
            }
        
        for tr in results['trainrounds']:
            print(tr.find('testrun'), tr)
            if tr.find('testrun') != -1:
                continue

            tr_result = {}
            if os.path.exists(tr):
                with open(tr, 'r') as f:
                    tr_result = json.load(f)
                
                filename = os.path.basename(tr)
                modelname = filename[:filename.find('_skills')]
                traindate = filename[filename.find('.stats')-8:filename.find('.stats')]

                lastEpochStr = str(tr_result['best_epoch'])
                bestepoch = tr_result['f1_scores'][lastEpochStr]
                # bestepoch = lastepoch - patience

                totalAccuraciesLastEpoch = [class_report['accuracy'] for class_report in tr_result["classification_reports"][lastEpochStr].values()]
                totalAccuracy = sum(totalAccuraciesLastEpoch) / len(totalAccuraciesLastEpoch)

                results[modelname][traindate] = {
                    'f1-scores-val' : tr_result['f1_scores'],
                    'f1-scores-val-total': [tr_result['f1_scores'][str(i)]['Total'] for i in range(len(tr_result['f1_scores']))],
                    'f1-scores-val-skill': [tr_result['f1_scores'][str(i)]['Skill'] for i in range(len(tr_result['f1_scores']))],
                    'f1-macro-avg' : tr_result['total_accuracy_at_best'],
                    'f1-macro-avg-skills' : bestepoch['Skill'],
                    'total-accuracy' : totalAccuracy,
                }

                if tr_result['total_accuracy_at_best'] > results[modelname]['best']['f1-macro-avg']:
                    results[modelname]['best'] = results[modelname][traindate]
                    results[modelname]['date'] = traindate
                    results['modelcomparison'][modelname] = {
                        'model': modelname,
                        'f1-macro-avg': round(100 * tr_result['total_accuracy_at_best'], 2),
                        'f1-macro-avg-skills' : round(100 * bestepoch['Skill'], 2),
                        'total-accuracy' : round(100 * totalAccuracy, 2),
                        'date' : traindate
                    }

                if tr_result['total_accuracy_at_best'] > results['best']['f1-macro-avg']:
                    modelcomparison = results['modelcomparison']
                    results['best'] = results[modelname][traindate]
                    results['modelcomparison'] = modelcomparison
                    results['best']['date'] = traindate
                    results['best']['modelname'] = modelname


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
            if os.path.exists(argpath):
                with open(argpath, 'r') as file:
                    results[traindir]['args'] = yaml.safe_load(file)

            # Process results
            if 'weights' not in results[traindir]['content']:
                continue

            if 'results.csv' not in results[traindir]['content']:
                continue

            df_results = pd.read_csv(os.path.join(trainpath, 'results.csv'))


        # checkpoint = torch.load(os.path.join('..', 'computervision', 'weights', f"{selectedModel}.checkpoint.pt"), weights_only=False)


        return results
