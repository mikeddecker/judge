import os
import json
import yaml
import pandas as pd
from .videoService import VideoService # TODO : move to repo
from .jobService import JobService # TODO : move to repo
from helpers.ConfigHelper import recognition_get_modelpaths, PYTORCH_MODELS_SKILLS, get_discipline_DoubleDutch_config
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.statsRepo import StatsRepository
from typing import List

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
        "StatsRepo",
        "videoService",
        "jobService",
    ]
    def __init__(self, videoService:VideoService):
        self.FolderRepo = FolderRepository(db=db)
        self.VideoRepo = VideoRepository(db=db)
        self.StatsRepo = StatsRepository(db=db)
        self.videoService = videoService
        self.jobService = JobService()
        
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
                freq_table[min(lvl, 8)] += 1
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

        for videoId in videoIds:
            scores[videoId] = {}
            scores[videoId]["videoId"] = videoId
            scores[videoId]["judges"] = self.videoService.get(id=videoId).JudgeDiffScore
            if scores[videoId]["judges"]:
                scores["total"]["judges"] += scores[videoId]["judges"]
            else:
                continue

            for model in allowed_models:

                # TODO : refactor
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
            if tr.find('testrun') != -1:
                continue

            tr_result = {}
            if os.path.exists(tr):
                with open(tr, 'r') as f:
                    tr_result = json.load(f)
                
                filename = os.path.basename(tr)
                modelname = filename[:filename.find('_skills')]

                lastEpochStr = str(tr_result['best_epoch'])

                totalAccuraciesLastEpoch = [class_report['accuracy'] for class_report in tr_result["classification_reports"][lastEpochStr].values()]
                totalAccuracy = sum(totalAccuraciesLastEpoch) / len(totalAccuraciesLastEpoch)
                totalWeightedF1LastEpoch = [class_report['weighted avg']['f1-score'] for class_report in tr_result["classification_reports"][lastEpochStr].values()]
                totalWeightedF1 = sum(totalWeightedF1LastEpoch) / len(totalWeightedF1LastEpoch)

                f1_macro_avg = 0
                # TODO : temp solution after delete
                if 'total_accuracy_at_best' in tr_result.keys():
                    f1_macro_avg = tr_result['total_accuracy_at_best']
                else:
                    f1_macro_avg = tr_result['f1_macro_avg_accuracy']
                
                results[modelname] = tr_result
                if modelname != 'best':
                    results['modelcomparison'][modelname] = {
                        "model" : modelname,
                        "f1-macro-avg" : f1_macro_avg,
                        "f1-macro-avg-skills" : tr_result['classification_reports'][lastEpochStr]['Skill']['macro avg']['f1-score'],
                        "f1-weighted-avg" : totalWeightedF1,
                        "f1-weighted-avg-skills" : tr_result['classification_reports'][lastEpochStr]['Skill']['weighted avg']['f1-score'],
                        "total-accuracy" : totalAccuracy,
                        "date" : 'pre-juli' if 'rundate' not in tr_result.keys() else tr_result['rundate'],
                        "hours_training" : 0 if 'time' not in tr_result.keys() else round(tr_result["time"] / 36) / 100
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
        
        return results

    def getLocalizeResults(self):
        results = {
            'boxcounts': {
                'total' : self.StatsRepo.localize_framelabels_total(),
                'daily' : self.StatsRepo.localize_framelabels_daily(),
            },
            'general' : {},
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

    def getGeneralStats(self) -> dict:
        return {
            'videos' : self.videoService.count()
        }