from managers.TrainerSkills import TrainerSkills
from constants import PYTORCH_MODELS_SKILLS

class Trainer:
    def train(self, type, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, modelparams: dict = None):
        match type:
            case 'LOCALIZE':
                raise NotImplementedError()
            case 'SEGMENT':
                raise NotImplementedError()
            case 'SKILL':
                if modelname in PYTORCH_MODELS_SKILLS.keys():
                    Trainer.__train_skills_pytorch(modelname=modelname,
                                                   from_scratch=from_scratch,
                                                   epochs=epochs,
                                                   save_anyway=save_anyway,
                                                   unfreeze_all_layers=unfreeze_all_layers,
                                                   trainparams=modelparams)
                else:
                    raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")

    def predict(modelname, videoId):
        pass

    def __train_skills_pytorch(modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, trainparams: dict = None):
        SkillTrainer = TrainerSkills()
        SkillTrainer.train(modelname=modelname,
                           from_scratch=from_scratch,
                           epochs=epochs,
                           save_anyway=save_anyway,
                           unfreeze_all_layers=unfreeze_all_layers,
                           trainparams = trainparams)
        

if __name__ == "__main__":
    trainparams = {
        "dim" : 224,
        "timesteps" : 16,
        "batch_size" : 1,
    }
    modelname = "HAR_SA_Conv3D"
    trainer = Trainer()
    trainer.train(
        type="SKILL",
        modelname=modelname,
        from_scratch=False,
        epochs=5,
        save_anyway=True,
        unfreeze_all_layers=False,
        modelparams=trainparams
    )