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

if __name__ == "__main__":
    trainparams = {
        "dim" : 224,
        "timesteps" : 16,
        "batch_size" : 1,
    }
    modelname = "HAR_SA_Conv3D"
    modelname = "HAR_MViT"
    trainer = Trainer()
    max_rounds = [2, 20]
    type="SEGMENT"
    
    
    ################################    
    trainer.train(
        type=type,
        modelname=modelname,
        from_scratch=True,
        epochs=max_rounds[0],
        save_anyway=True,
        unfreeze_all_layers=False,
        modelparams=trainparams,
        learning_rate=4e-5
    )

    trainer.train(
        type=type,
        modelname=modelname,
        from_scratch=False,
        epochs=max_rounds[1],
        save_anyway=True,
        unfreeze_all_layers=True,
        modelparams=trainparams,
        learning_rate=1e-6
    )