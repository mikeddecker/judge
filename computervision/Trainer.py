from managers.TrainerSkills import TrainerSkills
from managers.TrainerSegments import TrainerSegments
from constants import PYTORCH_MODELS_SKILLS

class Trainer:
    def train(self, type, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, modelparams: dict = None, learning_rate=1e-5):
        match type:
            case 'LOCALIZE':
                raise NotImplementedError()
            case 'SEGMENT':
                segmentTrainer = TrainerSegments()
                segmentTrainer.train(
                    modelname=modelname,
                    from_scratch=from_scratch,
                    epochs=epochs,
                    save_anyway=save_anyway,
                    unfreeze_all_layers=unfreeze_all_layers,
                    trainparams=modelparams,
                    learning_rate=learning_rate
                )
            case 'SKILL':
                if modelname in PYTORCH_MODELS_SKILLS.keys():
                    skillTrainer = TrainerSkills()
                    skillTrainer.train(modelname=modelname,
                                                   from_scratch=from_scratch,
                                                   epochs=epochs,
                                                   save_anyway=save_anyway,
                                                   unfreeze_all_layers=unfreeze_all_layers,
                                                   trainparams=modelparams,
                                                   learning_rate=learning_rate)
                else:
                    raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")


trainparams = {}
mvitparams =  {    
    "balancedType" : "limit_10procent",
    "dim" : 224,
    "timesteps" : 16,
    "batch_size" : 1,
}
resnetparams = {
    "balancedType" : "limit_10procent",
    "dim" : 224, # 112 originally
    "timesteps" : 16,
    "batch_size" : 1,
}
swinparams =  {    
    "balancedType" : "limit_10procent",
    "dim" : 224,
    "timesteps" : 16,
    "batch_size" : 1,
}

trainparams["HAR_MViT"] = mvitparams
trainparams["HAR_MViT_extra_dense"] = mvitparams
trainparams["HAR_SA_Conv3D"] = mvitparams
trainparams["HAR_Resnet_R3D"] = resnetparams
trainparams["HAR_Resnet_MC3"] = resnetparams
trainparams["HAR_Resnet_R2plus1"] = resnetparams
trainparams["HAR_SwinT_t"] = swinparams
trainparams["HAR_SwinT_s"] = swinparams

modelname = "HAR_SA_Conv3D"
modelname = "HAR_MViT"
max_rounds = [4, 60]

models = [
    'HAR_MViT',
    'HAR_Resnet_MC3',
    # 'HAR_SA_Conv3D',
    'HAR_Resnet_R2plus1',
    'HAR_SwinT_t',
    'HAR_SwinT_s',
    'HAR_Resnet_R3D',
    'HAR_MViT_extra_dense',
]
if __name__ == "__main__":
    trainer = Trainer()
    

    for modelname in models:
        print("@"*80)
        print(modelname)
        print("@"*80)
        ################################    
        # trainer.train(
        #     type="SEGMENT",
        #     modelname=modelname,
        #     from_scratch=True,
        #     epochs=max_rounds[0],
        #     save_anyway=True,
        #     unfreeze_all_layers=False,
        #     modelparams=trainparams,
        #     learning_rate=4e-5
        # )

        # trainer.train(
        #     type="SEGMENT",
        #     modelname=modelname,
        #     from_scratch=False,
        #     epochs=max_rounds[1],
        #     save_anyway=True,
        #     unfreeze_all_layers=True,
        #     modelparams=trainparams,
        #     learning_rate=1e-6
        # )

        trainer.train(
            type="SKILL",
            modelname=modelname,
            from_scratch=True,
            epochs=max_rounds[0],
            save_anyway=True,
            unfreeze_all_layers=False,
            modelparams=trainparams[modelname],
            learning_rate=3e-5
        )

        trainer.train(
            type="SKILL",
            modelname=modelname,
            from_scratch=False,
            epochs=max_rounds[1],
            save_anyway=True,
            unfreeze_all_layers=True,
            modelparams=trainparams[modelname],
            learning_rate=8e-7
        )