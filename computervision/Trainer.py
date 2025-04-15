from managers.TrainerSkills import TrainerSkills
from constants import PYTORCH_MODELS_SKILLS

class Trainer:
    def train(self, type, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, modelparams: dict = None, learning_rate=1e-5):
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
                                                   trainparams=modelparams,
                                                   learning_rate=learning_rate)
                else:
                    raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")

    def predict(modelname, videoId):
        pass

    def __train_skills_pytorch(modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, trainparams: dict = None, learning_rate=1e-5):
        SkillTrainer = TrainerSkills()
        SkillTrainer.train(modelname=modelname,
                           from_scratch=from_scratch,
                           epochs=epochs,
                           save_anyway=save_anyway,
                           unfreeze_all_layers=unfreeze_all_layers,
                           trainparams = trainparams,
                           learning_rate=learning_rate)
        

if __name__ == "__main__":
    trainparams = {
        "dim" : 224,
        "timesteps" : 16,
        "batch_size" : 1,
    }
    modelname = "HAR_MViT"
    trainer = Trainer()
    trainer.train(
        type="SKILL",
        modelname=modelname,
        from_scratch=True,
        epochs=3,
        save_anyway=True,
        unfreeze_all_layers=True,
        modelparams=trainparams,
        learning_rate=4e-5
    )

    trainer.train(
        type="SKILL",
        modelname=modelname,
        from_scratch=False,
        epochs=4,
        save_anyway=True,
        unfreeze_all_layers=True,
        modelparams=trainparams,
        learning_rate=1e-6
    )