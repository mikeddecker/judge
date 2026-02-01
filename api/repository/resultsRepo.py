import os
from config import RECIPES, ENVS
from flask_sqlalchemy import SQLAlchemy
from helpers.helpers import load_json_file
from helpers.ConfigHelper import recognition_get_modelpaths
from repository.models import TrainResult
from sqlalchemy import or_

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
        # TODO : filter skill only
        query = self.db.session.query(
            TrainResult,
        ).filter_by(
            isTestrun = False
        ).filter(
            or_(
                TrainResult.isBestOfAll == True,
                TrainResult.isBestOfArchitecture == True,
                TrainResult.isBestOfRecipe == True
            )
        )

        result = query.all()

        return [tr.to_dict() for tr in result]

    def judge(self) -> dict:
        return {}

