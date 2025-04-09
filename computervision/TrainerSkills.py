
KERAS_MODELS = []
PYTORCH_MODELS = []

class TrainerSkills:
    def train(modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False):
        pass

    def predict(modelname, videoId):
        pass

    def __addPytorchTop(model):
        """Returns a given pytorch model with the skill top predictions"""
        raise NotImplementedError()

    def __addKerasTop(model):
        """Returns a given keras model with the skill top predictions"""
        raise NotImplementedError()