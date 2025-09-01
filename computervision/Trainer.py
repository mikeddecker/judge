from managers.TrainerSkills import TrainerSkills
from managers.TrainerSegments import TrainerSegments
from managers.TrainerLocalize import train_yolo_model, validate_localize
from managers.DataRepository import DataRepository
from constants import PYTORCH_MODELS_SKILLS
from constants import RECIPES, SPEEDMODES

class Trainer:
    def train(self, step, recipename, from_scratch, save_anyway, speedmode=SPEEDMODES[0]):
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
                if modelname in PYTORCH_MODELS_SKILLS.keys():
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

trainparams = {}
mvitparams =  {    
    "balancedType" : "limit_5procent",
    "dim" : 224,
    "timesteps" : 16,
    "batch_size" : 1,
}
resnetparams = {
    "balancedType" : "limit_5procent",
    "dim" : 224, # 112 originally
    "timesteps" : 16,
    "batch_size" : 1,
}
swinparams =  {    
    "balancedType" : "limit_5procent",
    "dim" : 224,
    "timesteps" : 16,
    "batch_size" : 1,
}

trainparams["MViT"] = mvitparams
trainparams["MViT_extra_dense"] = mvitparams
trainparams["SA_Conv3D"] = mvitparams
trainparams["Resnet_R3D"] = resnetparams
trainparams["Resnet_MC3"] = resnetparams
trainparams["Resnet_R2plus1"] = resnetparams
trainparams["SwinT_t"] = swinparams
trainparams["SwinT_s"] = swinparams

modelname = "SA_Conv3D"
modelname = "MViT"
max_rounds = [4, 60]

models = [
    'MViT',
    'Resnet_MC3',
    # 'SA_Conv3D',
    'Resnet_R2plus1',
    'SwinT_t',
    'SwinT_s',
    'Resnet_R3D',
    'MViT_extra_dense',
]
if __name__ == "__main__":
    # trainer = Trainer()
    pass

