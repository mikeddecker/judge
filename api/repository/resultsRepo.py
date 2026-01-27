import os
from config import RECIPES, ENVS
from flask_sqlalchemy import SQLAlchemy
from helpers.helpers import load_json_file
from helpers.ConfigHelper import recognition_get_modelpaths

def extract_key_number_pairs(obj):
    if isinstance(obj, list):
        for item in obj:
            yield from extract_key_number_pairs(item)
    else:
        for k, v in obj.items():
            if isinstance(v, (int, float)):
                yield (k, v)
            elif isinstance(v, (dict, list)):
                yield from extract_key_number_pairs(v)

class ResultsRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db

    def general(self) -> dict:
        return {}

    def localization(self) -> dict:
        results = {}
        for key, recipe in RECIPES['LOCALIZE'].items():
            resultdir = os.path.join(ENVS.DIRS.WEIGHTS.YOLO, recipe.size)
            subfolder = os.listdir(resultdir)[0]
            resultdir = os.path.join(resultdir, subfolder)
            ious_all = load_json_file(os.path.join(resultdir, 'localize_ious.json'))
            recipe_results = load_json_file(os.path.join(resultdir, 'results.json'))
            if ious_all:
                results[key] = {
                    'model': key,
                    'team_raw_avg' : ious_all['raw']['val']['avg'],
                    'team_smoothing_avg' : ious_all['smoothing']['val']['avg'],
                    **recipe_results['results_dict'],
                    'ious': { **ious_all },
                }
        return results

    def segmentation(self) -> dict:
        return {}

    def recognition(self) -> dict:
        return {}
        model_results = {}
        model_paths =  recognition_get_modelpaths()

        for train_round_path in model_paths:
            if train_round_path.find('testrun') != -1:
                continue

            train_round_result = load_json_file(train_round_path)

            if not train_round_result:
                continue

            modelname = train_round_result['modelname'] if 'best' not in train_round_path else 'best'
            model_results[modelname] = train_round_result

        return model_results

    def judge(self) -> dict:
        return {}

