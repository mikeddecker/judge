class LayerComposition:
    def __init__(
            self,
            compositionName: str,
            generalProperties: dict, # Where stage = NULL
            startProperties: dict, # Where stage = 0
            endProperties: dict, # Where stage = 1
            stageProperties: dict, # Key = stage nr
        ):
        self.CompositionName = compositionName
        self.GeneralProperties = generalProperties
        self.StartProperties = startProperties
        self.EndProperties = endProperties
        self.StageProperties = stageProperties
    
    def to_dict(self):
        return {
            'compositionName' : self.CompositionName,
            'GeneralProperties' : self.GeneralProperties,
            'StartProperties' : self.StartProperties,
            'EndProperties' : self.EndProperties,
            'StageProperties' : self.StageProperties,
        }

