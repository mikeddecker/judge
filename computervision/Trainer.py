from managers.TrainerSkills import TrainerSkills
from managers.TrainerSegments import TrainerSegments
from managers.TrainerLocalize import train_yolo_model
from managers.RepoGeneral import REPO_GENERAL
from constants import PYTORCH_MODELS_SKILLS
from constants import RECIPES, SPEEDMODES

class Trainer:
    _testrun:bool=False
    def __init__(self, testrun:bool=False):
        self._testrun = testrun

    def train(self, step, recipename, speedmode=SPEEDMODES[1], job_arguments: dict = {}) -> None:
        match step:
            case 'LOCALIZE':
                train_yolo_model(RECIPES[step][recipename].size, repo=REPO_GENERAL)
            case 'SEGMENT':
                segmentTrainer = TrainerSegments(self._testrun)
                segmentTrainer.train(
                    modelname=RECIPES[step][recipename].model,
                    epochs=1,
                    unfreeze_all_layers=False,
                    trainparams=RECIPES[step][recipename],
                )
                segmentTrainer.train(
                    modelname=RECIPES[step][recipename].model,
                    epochs=300,
                    unfreeze_all_layers=True,
                    trainparams=RECIPES[step][recipename],
                )
            case 'SKILL':
                # recipename was modelname, prior to refactor
                if RECIPES[step][recipename].model in PYTORCH_MODELS_SKILLS.keys():
                    skillTrainer = TrainerSkills(self._testrun)
                    skillTrainer.train(
                        recipe=RECIPES[step][recipename],
                        speedmode=speedmode,
                        job_arguments=job_arguments
                    )
                else:
                    raise NotImplementedError('Unknown model', RECIPES[step][recipename].model)
            case _:
                raise ValueError(f"Trainer - Type {step} not recognized")

