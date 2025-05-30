import os
import torch
import glob
import json
import yaml
import pandas as pd
from .videoService import VideoService
from .jobService import JobService
from domain.folder import Folder
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from helpers.ValueHelper import ValueHelper
from typing import List
from helpers.ConfigHelper import recognition_get_modelpaths, PYTORCH_MODELS_SKILLS, get_discipline_DoubleDutch_config

LEVEL_TO_SCORE_MAP = {
    0 : 0,
    1 : 1.5,
    2 : 2.2,
    3 : 3.3,
    4 : 4.9,
    5 : 7.3,
    6 : 11,
    7 : 11,
    8 : 11,
}

class StatsService:
    PROPERTIES = [
        "FolderRepo",
        "VideoRepo",
        "StorageFolder",
        "videoService",
        "jobService",
    ]
    def __init__(self, storage_folder: str, videoService:VideoService):
        ValueHelper.check_raise_string(storage_folder)
        self.FolderRepo = FolderRepository(db=db)
        self.VideoRepo = VideoRepository(db=db)
        self.videoService = videoService
        self.jobService = JobService(storage_folder=storage_folder)

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

    def __calculate_diff_score(self, videoId: int, model: str):
        freq_table = {l: 0 for l in range(9)}

        predicted_skills = self.videoService.load_predicted_skills(videoId=videoId, model=model)
        
        config = get_discipline_DoubleDutch_config()
        levels = [ 
            self.videoService.calculate_skill_level(
                disciplineconfig=config,
                skillinfo= {k: v['y_pred'] if config[k][0] == "Categorical" else v['y_pred'] for k, v in predicted_skills[frameStart].items()},
                frameStart=int(frameStart),
                videoId=videoId
            ) for frameStart in
            predicted_skills.keys()
        ]
        levels = [lvl if not isinstance(lvl, list) else lvl[0] for lvl in levels]
        
        score = 0
        for lvl in levels:
            if isinstance(lvl, int):
                freq_table[lvl] += 1
                score += LEVEL_TO_SCORE_MAP[min(lvl, 8)]
            else:
                pass # Mistakes

        return freq_table, score
    
    def get_score_comparison(self, videoIds: List[int]):
        allowed_models = PYTORCH_MODELS_SKILLS.keys()
        scores = { 
            'total' : { m: 0 for m in allowed_models }
        }
        scores['total']['judges'] = 0

        recognition_results = self.getRecognitionResults('')
        

        for videoId in videoIds:
            scores[videoId] = {}
            scores[videoId]["videoId"] = videoId
            scores[videoId]["judges"] = self.videoService.get(id=videoId).JudgeDiffScore
            if scores[videoId]["judges"]:
                scores["total"]["judges"] += scores[videoId]["judges"]

            for model in allowed_models:
                if model not in allowed_models:
                    return f"Model {model} not allowed", 404

                if self.videoService.video_has_predictions(videoId=videoId, model=model): # and not self.jobService.video_has_pending_job(videoId=videoId, model=model):
                    # TODO : add re-calculate after x days or when a new model has been trained
                    freq, score = self.__calculate_diff_score(videoId=videoId, model=model)
                    
                    scores[videoId][model] = round(score, 2)
                    scores[videoId][f"{model}_freq"] = freq
                    
                    if scores[videoId]["judges"]:
                        scores[videoId][f"{model}_procent_difference"] = round(100 * (scores[videoId][model] - scores[videoId]["judges"]) / scores[videoId]["judges"], 2)
                        scores["total"][model] += round(score, 2)

                # elif not self.jobService.video_has_pending_job(videoId=videoId, model=model): # TODO : optimize query!!
                #     self.jobService.launch_job_predict_skills(step='FULL', model=model, videoId=videoId)
                #     scores[videoId][model] = "Created"
                else:
                    scores[videoId][model] = "Waiting"


        if scores['total']["judges"]:
            scores["total"]["judges"] = round(scores["total"]["judges"], 2)

        for model in allowed_models:
            if scores["total"][model]:
                scores["total"][f"{model}_procent_difference"] = round(100 * (scores["total"][model] - scores["total"]["judges"]) / scores["total"]["judges"], 2)

        return scores


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
                totalWeightedF1LastEpoch = [class_report['weighted avg']['f1-score'] for class_report in tr_result["classification_reports"][lastEpochStr].values()]
                totalWeightedF1 = sum(totalWeightedF1LastEpoch) / len(totalWeightedF1LastEpoch)

                results[modelname][traindate] = {
                    'f1-scores-val' : tr_result['f1_scores'],
                    'f1-scores-val-total': [tr_result['f1_scores'][str(i)]['Total'] for i in range(len(tr_result['f1_scores']))],
                    'f1-scores-val-skill': [tr_result['f1_scores'][str(i)]['Skill'] for i in range(len(tr_result['f1_scores']))],
                    'f1-macro-avg' : tr_result['total_accuracy_at_best'],
                    'f1-macro-avg-skills' : bestepoch['Skill'],
                    'f1-weighted-avg' : totalWeightedF1,
                    'f1-weighted-avg-skills' : tr_result["classification_reports"][lastEpochStr]['Skill']['weighted avg']['f1-score'],
                    'total-accuracy' : totalAccuracy,
                }

                if tr_result['total_accuracy_at_best'] > results[modelname]['best']['f1-macro-avg']:
                    results[modelname]['best'] = results[modelname][traindate]
                    results[modelname]['date'] = traindate
                    results['modelcomparison'][modelname] = {
                        'model': modelname,
                        'f1-macro-avg': round(100 * tr_result['total_accuracy_at_best'], 2),
                        'f1-macro-avg-skills' : round(100 * bestepoch['Skill'], 2),
                        'f1-weighted-avg': round(100 * totalWeightedF1, 2),
                        'f1-weighted-avg-skills' : round(100 * tr_result["classification_reports"][lastEpochStr]['Skill']['weighted avg']['f1-score'], 2),
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
