# Imports

import time
import json
from managers.DataRepository import DataRepository
from Predictor import Predictor, modelparams
from Trainer import Trainer, trainparams, max_rounds
from TrainerLocalize import train_yolo_model, validate_localize

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

    if job["type"] == "PREDICT":
        print(job)
        job_arguments = json.loads(job["job_arguments"])
        saveAsMp4 = False if "save_mp4" not in job_arguments.keys() else job_arguments["save_mp4"]
        predictor.predict(
            type=job["step"],
            videoId=job_arguments["videoId"],
            modelname=job_arguments["model"],
            modelparams=modelparams[job_arguments["model"]],
            saveAsVideo=saveAsMp4,
        )
        REPO.delete_job(job["id"])
    elif job["type"] == "TRAIN":
        print(job)

        # LOCALIZE: TODO : wrap in trainer
        size = 'n'
        variant = f'yolo11{size}.pt'
        save_dir = train_yolo_model(variant=variant, repo=REPO)
        modelname = f"yolov11{size}_{save_dir.name}"
        validate_localize(modeldir=save_dir, repo=REPO, modelname=modelname)

        # Create videocrops TODO: move to datagenerators or remove freshly labeled videos
        for videoId in REPO.get_videoIds_of_videos_with_skills():
            # TODO : include modelname in crop video, but also a sort of confidence score.
            predictor.predict(
                type="LOCALIZE",
                videoId=videoId,
                modelname=None,
                saveAsVideo=True
            )

        print("Start training segments")

        models = [
            'HAR_MViT',
            'HAR_Resnet_MC3',
            'HAR_SA_Conv3D',
            'HAR_Resnet_R2plus1',
            'HAR_Resnet_R3D',
            'HAR_MViT_extra_dense',
        ]

        for modelname in models:
            # modelname = 'HAR_MViT' # TODO : pick from job executor
            # trainer.train(
            #     type="SEGMENT",
            #     modelname=modelname,
            #     from_scratch=True,
            #     epochs=max_rounds[0],
            #     save_anyway=True,
            #     unfreeze_all_layers=False,
            #     modelparams=trainparams[modelname],
            #     learning_rate=4e-5
            # )

            trainer.train(
                type="SEGMENT",
                modelname=modelname,
                from_scratch=False,
                epochs=max_rounds[1],
                save_anyway=True,
                unfreeze_all_layers=True,
                modelparams=trainparams[modelname],
                learning_rate=1e-6
            )

            trainer.train(
                type="SKILL",
                modelname=modelname,
                from_scratch=True,
                epochs=max_rounds[0],
                save_anyway=True,
                unfreeze_all_layers=False,
                modelparams=trainparams[modelname],
                learning_rate=4e-5
            )

            trainer.train(
                type="SKILL",
                modelname=modelname,
                from_scratch=False,
                epochs=max_rounds[1],
                save_anyway=True,
                unfreeze_all_layers=True,
                modelparams=trainparams[modelname],
                learning_rate=1e-6
            )
        
        REPO.check_connection_reconnect_if_needed()
        REPO.delete_job(job["id"])
    else:
        print('Unrecognized job?')
        print(job)
        time.sleep(2)
    # Update, remove job
    




