# Imports

import time
import json
from managers.DataRepository import DataRepository
from Predictor import Predictor
from Trainer import Trainer, trainparams, max_rounds
from managers.TrainerLocalize import train_yolo_model, validate_localize
from constants import RECIPES
from pprint import pprint

# Managers

REPO = DataRepository()

# JobReader

no_shutdown_job = True
predictor = Predictor()
trainer = Trainer()


while no_shutdown_job:
    job = REPO.get_next_job()

    if job is None:
        print('Waiting for a job')
        time.sleep(3)
        continue
    else:
        print(job)
        pprint(job["job_arguments"])

    job_arguments = json.loads(job["job_arguments"])
    if job["type"] == "PREDICT":
        saveAsMp4 = False if "save_mp4" not in job_arguments.keys() else bool(job_arguments["save_mp4"])
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

    
        REPO.delete_job(job["id"])
    else:
        print('Unrecognized job?')
        print(job)
        time.sleep(2)
    # Update, remove job
    




