import os
import yaml
import glob

PYTORCH_MODELS_SKILLS = {
    "HAR_SA_Conv3D" : None,
    "HAR_MViT" : None,
    "HAR_MViT_extra_dense" : None,
    "HAR_Resnet_R3D" : None,
    "HAR_Resnet_MC3" : None,
    "HAR_Resnet_R2plus1" : None,
    "HAR_SwinT_t" : None,
    "HAR_SwinT_s" : None,
}

def get_discipline_DoubleDutch_config(include_tablename=True):
    config = {
        "Type" : ("Categorical", "Type"), # Will be textual representions
        "Rotations" : ("Numerical", 0, 8, 1), # min, max, step
        "Turner1": ("Categorical", "Turner"),
        "Turner2": ("Categorical", "Turner"),
        "Skill" : ("Categorical", "Skill"),
        "Hands" : ("Numerical", 0, 2, 1), # 0 for al salto types
        "Feet" : ("Numerical", 0, 2, 1),
        "Turntable" : ("Numerical", 0, 4, 0.25), # Per quarter (but still integers)
        "BodyRotations" : ("Numerical", 0, 2, 0.5),
        "Backwards" : ("Boolean"),
        "Sloppy" : ("Boolean"),
        "Hard2see" : ("Boolean"),
        "Fault" : ("Boolean"),
    }
    if include_tablename:
        config["Tablename"] = "DoubleDutch"
    return config

def localize_get_best_modelpath():
    """Returns modelname, modelpath, e.g. yolo11n ./runs/detect/train7"""
    # TODO : update to take actual best
    folder_path = os.path.join('..', 'runs', 'detect')
    trainrounds = os.listdir(folder_path)
    nrs = [0 if s[5:] == '' else int(s[5:]) for s in trainrounds]
    maxround = max(nrs)
    if maxround == 0:
        maxround == ''

    modelpath = os.path.join(folder_path, f"train{maxround}")

    argpath = os.path.join(modelpath, 'args.yaml')
    modelname = 'pathDoesNotExist'
    if os.path.exists(argpath):
        with open(argpath, 'r') as file:
            modelname = yaml.safe_load(file)['model'].split('.')[0]
    
    return modelname, modelpath


def recognition_get_modelpaths():
    """Returns path to modelstats, e.g. ./weights/HAR_MViT_skills_20250524.stats.json"""
    # TODO : update to take actual best
    folder_path = os.path.join('..', 'computervision', 'weights', f"*skills*.stats.json")
    trainrounds = glob.glob(folder_path)
        
    return trainrounds
