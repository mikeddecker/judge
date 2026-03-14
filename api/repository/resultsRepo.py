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
        try:
            for key, recipe in RECIPES['LOCALIZE'].items():
                try:
                    resultdir = os.path.join(ENVS.DIRS.WEIGHTS.YOLO, recipe.size)
                    
                    # Check if directory exists
                    if not os.path.exists(resultdir):
                        print(f"⚠️ Localization results directory not found: {resultdir}")
                        continue
                    
                    # Get first subdirectory
                    subfolders = os.listdir(resultdir)
                    if not subfolders:
                        print(f"⚠️ No subdirectories in localization results: {resultdir}")
                        continue
                    
                    subfolder = subfolders[0]
                    resultdir = os.path.join(resultdir, subfolder)
                    
                    # Load JSON files
                    ious_file = os.path.join(resultdir, 'localize_ious.json')
                    results_file = os.path.join(resultdir, 'results.json')
                    
                    if not os.path.exists(ious_file):
                        print(f"⚠️ IOUs file not found: {ious_file}")
                        continue
                    
                    if not os.path.exists(results_file):
                        print(f"⚠️ Results file not found: {results_file}")
                        continue
                    
                    ious_all = load_json_file(ious_file)
                    recipe_results = load_json_file(results_file)
                    
                    if ious_all and recipe_results:
                        results[key] = {
                            'model': key,
                            'team_raw_avg' : ious_all.get('raw', {}).get('val', {}).get('avg', 0),
                            'team_smoothing_avg' : ious_all.get('smoothing', {}).get('val', {}).get('avg', 0),
                            **(recipe_results.get('results_dict', {})),
                            'ious': { **ious_all },
                        }
                except Exception as e:
                    print(f"⚠️ Error loading localization results for {key}: {e}")
                    continue
        except Exception as e:
            print(f"❌ Error in localization results: {e}")
        
        return results

    def segmentation(self) -> dict:
        # Return empty dict - segmentation results not yet implemented
        return {}

    def recognition(self) -> dict:
        # Safely load recognition results
        try:
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
            return [tr.to_dict() for tr in result] if result else []
        except Exception as e:
            print(f"❌ Error loading recognition results: {e}")
            return []

    def judge(self) -> dict:
        # Return empty dict - judge results not yet implemented
        return {}

