# Imports

import time
import json
from uuid import UUID
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
        # Convert videoId from string to UUID object
        videoId = UUID(job_arguments["videoId"]) if isinstance(job_arguments["videoId"], str) else job_arguments["videoId"]
        print(f"Predict video {videoId}")
        predictor = Predictor()
        predictor.predict(
            type=job["step"],
            videoId=videoId,
            recipename=job_arguments["model"],
            modelparams=RECIPES[job["step"]][job_arguments["model"]],
            saveAsVideo=False if "save_mp4" not in job_arguments.keys() else bool(job_arguments["save_mp4"]),
            weights=job_arguments["weights"] if job_arguments["weights"] is not None else 'best',
        )
        # Convert job id from bytes to UUID object
        jobId = UUID(bytes=job["id"]) if isinstance(job["id"], bytes) else job["id"]
        REPO_GENERAL.delete_job(jobId)
    elif job["type"] == "TRAIN":
        trainer = Trainer(testrun=job_arguments.get('testrun'))
        trainer.train(
            step=job['step'],
            recipename=job_arguments['recipe'],
            job_arguments=job_arguments
        )
        # Convert job id from bytes to UUID object
        jobId = UUID(bytes=job["id"]) if isinstance(job["id"], bytes) else job["id"]
        REPO_GENERAL.delete_job(jobId)
    else:
        print('Unrecognized job?')
        print(job)
        time.sleep(2)
    # Update, remove job

