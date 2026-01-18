# Imports

import time
import json
from managers.RepoGeneral import REPO_GENERAL
from Predictor import Predictor
from Trainer import Trainer
from sqlalchemy.orm import close_all_sessions
from constants import RECIPES
from pprint import pprint

# JobReader

no_shutdown_job = True

while no_shutdown_job:
    job = REPO_GENERAL.get_next_job()

    if job is None:
        time.sleep(3)
        close_all_sessions()
        continue
    else:
        print(f"Processing job ID: {job['id']}, Type: {job['type']}, Step: {job['step']}")
        job_arguments = json.loads(job["job_arguments"])
        print("Job arguments:")
        pprint(job_arguments)

    job_arguments : dict = json.loads(job["job_arguments"])

    if job["type"] == "PREDICT":
        print(f"Predict video {job_arguments["videoId"]}")
        predictor = Predictor(job_arguments.get('testrun'))
        predictor.predict(
            type=job["step"],
            videoId=job_arguments["videoId"],
            recipename=job_arguments["model"],
            modelparams=RECIPES[job["step"]][job_arguments["model"]],
            saveAsVideo=False if "save_mp4" not in job_arguments.keys() else bool(job_arguments["save_mp4"]),
            weights=job_arguments["weights"] if job_arguments["weights"] is not None else 'best',
            testrun=job_arguments['testrun']
        )
        REPO_GENERAL.delete_job(job["id"])
    elif job["type"] == "TRAIN":
        trainer = Trainer(testrun=job_arguments.get('testrun'))
        trainer.train(
            step=job['step'],
            recipename=job_arguments['recipe'],
            job_arguments=job_arguments
        )
        REPO_GENERAL.delete_job(job["id"])
    else:
        print('Unrecognized job?')
        print(job)
        time.sleep(2)
    # Update, remove job

