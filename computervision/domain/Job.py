class Job:
    Type = None # LOCALIZATION, SEGMENTATION, SKILL_RECOGNITION
    ModelName = None # e.g. yolo_v11_ultralitics or HAR_SA_Conv3D_medium_strides_skip

    def __init__(self, type, modelname, date):
        self.Type = type
        self.ModelName = modelname

    