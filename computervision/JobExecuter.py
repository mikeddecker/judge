# Imports

import time
import json
from managers.DataRepository import DataRepository
from Predictor import Predictor, modelparams

# Managers

REPO = DataRepository()


# JobReader

no_shutdown_job = True
predictor = Predictor()

while no_shutdown_job:
    job = REPO.get_next_job()

    if job is None:
        time.sleep(3)
        continue

    if job["type"] == "PREDICT":
        print(job)
        job_arguments = json.loads(job["job_arguments"])
        predictor.predict(
            type=job["step"],
            videoId=job_arguments["videoId"],
            modelname=job_arguments["model"],
            modelparams=modelparams[job_arguments["model"]],
            saveAsVideo=False,
        )
        REPO.delete_job(job["id"])
    else:
        time.sleep(5)
    # Update, remove job
    




