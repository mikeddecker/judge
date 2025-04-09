from TrainerSkills import TrainerSkills

KERAS_MODELS = []
PYTORCH_MODELS = []

class Trainer:
    def train(self, type, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, modelparams: dict = None):
        match type:
            case 'LOCALIZE':
                raise NotImplementedError()
            case 'SEGMENT':
                raise NotImplementedError()
            case 'SKILL':
                Trainer.__train_skills
                raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")

    def predict(modelname, videoId):
        pass

    def __train_skills(modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, modelparams: dict = None):
        pass