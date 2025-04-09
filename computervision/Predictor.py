from TrainerSkills import TrainerSkills

KERAS_MODELS = []
PYTORCH_MODELS = []

class Predictor:
    def predict(self, type: str, modelname: str, videoId: int, use_cropped_video_if_exists=True, save_predictions=True):
        match type:
            case 'LOCALIZE':
                raise NotImplementedError()
            case 'SEGMENT':
                raise NotImplementedError()
            case 'SKILL':
                raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")


    def __train_skills(modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, modelparams: dict = None):
        pass