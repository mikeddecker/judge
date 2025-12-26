# Imports

import time
import json
from collections import defaultdict
from managers.DataRepository import DataRepository
from Predictor import Predictor
from Trainer import Trainer
from sqlalchemy.orm import close_all_sessions
from constants import RECIPES
from pprint import pprint

# Managers

REPO = DataRepository()

# JobReader

no_shutdown_job = True
predictor = Predictor()
trainer = Trainer()

def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = dict(d)
    return {
        k: defaultdict_to_dict(v) if isinstance(v, (dict, defaultdict)) else v
        for k, v in d.items()
    }

def clean_skills(traintest: str):
    skills = REPO.get_skills(traintest)

    for index, skillrow in skills.iterrows():
        rowId = skillrow['id']
        if rowId > 1203:
            continue
        print("@"*40, rowId, "@"*40)
        cleaned_skill = {}
        uncleaned_skill : dict = skillrow['skillinfo']

        for composition_name, composition_properties in uncleaned_skill.items():
            cleaned_skill[composition_name] = []
            for composition_index, composition_stages in enumerate(composition_properties):
                cleaned_composition_stages = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                for stage, stageProperties in composition_stages.items():
                    if stage == 'compositionName':
                        continue
                    elif stage != 'StageProperties':
                        for property, value in stageProperties.items():
                            if property in ['Feet', 'RopesHolding'] and value != 2:
                                print(composition_name, composition_index, stage, property, value)
                                cleaned_composition_stages[stage][property] = value
                            elif property not in ['Feet', 'TurningBackwards', 'PointWorthy', 'RopesHolding'] and value != 0:
                                print(composition_name, composition_index, stage, property, value)
                                cleaned_composition_stages[stage][property] = value
                            elif (property == 'PointWorthy' and value != 'true') or (property == 'TurningBackwards' and value != 60):
                                cleaned_composition_stages[stage][property] = value
                            
                    else:
                        for numericStage, numericStageProperties in stageProperties.items():
                            for property, value in numericStageProperties.items():
                                if property in ['Feet', 'RopesHolding'] and value != 2:
                                    print(composition_name, composition_index, stage, numericStage, property, value)
                                    cleaned_composition_stages[stage][numericStage][property] = value
                                elif property not in ['Feet', 'TurningBackwards', 'PointWorthy', 'RopesHolding'] and value != 0:
                                    print(composition_name, composition_index, stage, numericStage, property, value)
                                    cleaned_composition_stages[stage][numericStage][property] = value
                                elif (property == 'PointWorthy' and value != 'true') or (property == 'TurningBackwards' and value != 60):
                                    cleaned_composition_stages[stage][numericStage][property] = value

                cleaned_skill[composition_name].append(defaultdict_to_dict(cleaned_composition_stages))

        REPO.update_skill(skillId=rowId, skillinfo=json.dumps(cleaned_skill))

while no_shutdown_job:
    job = REPO.get_next_job()

    if job is None:
        time.sleep(3)
        close_all_sessions()
        continue
    else:
        print(f"Processing job ID: {job['id']}, Type: {job['type']}, Step: {job['step']}")
        job_arguments = json.loads(job["job_arguments"])
        print("Job arguments:")
        pprint(job_arguments)

    job_arguments = json.loads(job["job_arguments"])
    if job["type"] == "PREDICT":
        saveAsMp4 = False if "save_mp4" not in job_arguments.keys() else bool(job_arguments["save_mp4"])
        print(f"Predict video {job_arguments["videoId"]}")
        predictor.predict(
            type=job["step"],
            videoId=job_arguments["videoId"],
            recipename=job_arguments["model"],
            modelparams=RECIPES[job["step"]][job_arguments["model"]],
            saveAsVideo=saveAsMp4,
            weights=job_arguments["weights"] if job_arguments["weights"] is not None else 'best'
        )
        REPO.delete_job(job["id"])
    elif job["type"] == "TRAIN":

        trainer.train(
            step=job['step'],
            recipename=job_arguments['recipe'],
            from_scratch=True,
            save_anyway=True,
        )
    elif job["type"] == "CLEAN":
        clean_skills('train')
        clean_skills('val')
        pass

        REPO.delete_job(job["id"])
    else:
        print('Unrecognized job?')
        print(job)
        time.sleep(2)
    # Update, remove job

