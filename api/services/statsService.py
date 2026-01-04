from .videoService import VideoService # TODO : move to repo
from .jobService import JobService # TODO : move to repo
from helpers.ConfigHelper import PYTORCH_MODELS_SKILLS, get_discipline_DoubleDutch_config
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
        layercomposition_names = self.StatsRepo.layercomposition_names()
        results = {
            'best' : {
                'f1-macro-avg' : 0
            },
            'selectedModel': selectedModel,
            'models' : self.StatsRepo.skills_metrics(),
            'modelcomparison' : {},
            'layerName_counts' : {
                'total': self.StatsRepo.skills_prop_counts(),
                **{ lcn: self.StatsRepo.skills_prop_counts(lcn) for lcn in layercomposition_names }
            },
            'prop_value_frequencies': self.StatsRepo.skills_prop_value_frequencies(),
            'skills': {
                'total' : self.StatsRepo.skill_counts(),
                'daily' : self.StatsRepo.skill_counts_daily(),
            },
            'layercomposition_names': layercomposition_names,
            'layercomposition_counts': self.StatsRepo.skill_counts_composition(),
        }
        print('selectedModel', selectedModel)

        return results

    def getLocalizeResults(self):
        results = {
            'boxcounts': {
                'total' : self.StatsRepo.localize_box_counts(),
                'daily' : self.StatsRepo.localize_box_counts_daily(),
            },
            'framecounts': {
                'total' : self.StatsRepo.localize_frame_counts(),
            },
            'recipes' : self.StatsRepo.localize_results(),
        }

        return results

    def getLocalizeLabelinfoPerVideo(self):
        return self.StatsRepo.get_localize_labelinfo_per_video()

    def getGeneralStats(self) -> dict:
        return {
            'videos' : self.videoService.count()
        }

