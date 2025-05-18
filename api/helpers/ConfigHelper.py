import os
import yaml

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
    print("|"*80)
    print(trainrounds)
    nrs = [0 if s[5:] == '' else int(s[5:]) for s in trainrounds]
    print(nrs)
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
