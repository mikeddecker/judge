from services.videoService import VideoService # TODO : move to repo
from services.jobService import JobService # TODO : move to repo
from config import LEVEL_TO_SCORE_MAP
from helpers.ConfigHelper import get_discipline_DoubleDutch_config, PYTORCH_MODELS_SKILLS
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.statsRepo import StatsRepository
from repository.resultsRepo import ResultsRepository
from typing import List
from uuid import UUID

class ResultsService:
    PROPERTIES = [
        "FolderRepo",
        "VideoRepo",
        "StatsRepo",
        "ResultsRepo",
        "videoService",
        "jobService",
    ]
    def __init__(self, videoService:VideoService):
        self.FolderRepo = FolderRepository(db=db)
        self.VideoRepo = VideoRepository(db=db)
        self.StatsRepo = StatsRepository(db=db)
        self.ResultsRepo = ResultsRepository(db=db)
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

    def general(self) -> dict:
        return self.ResultsRepo.general()

    def localization(self):
        return self.ResultsRepo.localization()

    def segmentation(self):
        return self.ResultsRepo.segmentation()

    def recognition(self):
        return self.ResultsRepo.recognition()

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

    def judge(self, videoIds: List[int]):
        # TODO : refactor to repo? Or partially?
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

