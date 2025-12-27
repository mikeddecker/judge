from managers.TrainerSkills import TrainerSkills
from managers.TrainerSegments import TrainerSegments
from managers.TrainerLocalize import train_yolo_model
from managers.DataRepository import DataRepository
from constants import PYTORCH_MODELS_SKILLS
from constants import RECIPES, SPEEDMODES

class Trainer:
    def train(self, step, recipename, from_scratch, save_anyway, speedmode=SPEEDMODES[0]) -> None:
        match step:
            case 'LOCALIZE':
                train_yolo_model(RECIPES[step][recipename].size, repo=DataRepository())
            case 'SEGMENT':
                segmentTrainer = TrainerSegments()
                segmentTrainer.train(
                    modelname=RECIPES[step][recipename].model,
                    from_scratch=from_scratch,
                    epochs=1,
                    save_anyway=save_anyway,
                    unfreeze_all_layers=False,
                    trainparams=RECIPES[step][recipename],
                )
                segmentTrainer.train(
                    modelname=RECIPES[step][recipename].model,
                    from_scratch=False,
                    epochs=300,
                    save_anyway=save_anyway,
                    unfreeze_all_layers=True,
                    trainparams=RECIPES[step][recipename],
                )
            case 'SKILL':
                # recipename was modelname, prior to refactor
                if recipename in PYTORCH_MODELS_SKILLS.keys():
                    skillTrainer = TrainerSkills()
                    skillTrainer.train(
                        recipe=RECIPES[step][recipename],
                        from_scratch=from_scratch,
                        epochs=300,
                        save_anyway=save_anyway,
                        unfreeze_all_layers=False,
                        speedmode=speedmode,
                        patience=1,
                    )
                    skillTrainer.train(
                        recipe=RECIPES[step][recipename],
                        from_scratch=False,
                        epochs=300,
                        save_anyway=save_anyway,
                        unfreeze_all_layers=True,
                        speedmode=SPEEDMODES[0]
                    )
                else:
                    raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {step} not recognized")

