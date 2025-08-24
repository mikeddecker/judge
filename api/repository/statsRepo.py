import json
import os
import re
from collections import Counter
from config import RECIPES, ENVS
from flask_sqlalchemy import SQLAlchemy
from helpers.helpers import load_json_file
from repository.models import Video as VideoInfoDB, Folder as FolderDB, FrameLabel, FrameLabelType
from repository.models import Skill, LayerComposition, LayerProperty, LayerPropertyValue
from sqlalchemy import desc, func, case, select, text
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

class StatsRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db
        self.split_train_test_framelabel = case(
            (FrameLabel.videoId % 10 == 5, 'test'),
            else_='train'
        ).label("split")
        self.split_train_test_skill = case(
            (Skill.videoId % 10 == 5, 'test'),
            else_='train'
        ).label("split")

    def localize_box_counts(self):
        counts = self.db.session.query(
            FrameLabel.labeltype,
            func.count().label("count"),
            self.split_train_test_framelabel
        ).group_by(
            FrameLabel.labeltype,
            self.split_train_test_framelabel
        ).all()

        return [
            {
                "type": row.labeltype,
                "count": row.count,
                "split": row.split
            }
            for row in counts
        ]
    
    def localize_frame_counts(self):
        # Step 1: Select distinct videoId + frameNr + split
        subq = self.db.session.query(
            FrameLabel.videoId,
            FrameLabel.frameNr,
            self.split_train_test_framelabel
        ).distinct(
            FrameLabel.videoId,
            FrameLabel.frameNr
        ).subquery()

        # Step 2: Count grouped by split in outer query
        counts = self.db.session.query(
            func.count().label("count"),
            subq.c.split
        ).group_by(
            subq.c.split
        ).all()
    
        return [
            {
                "count": row.count,
                "split": row.split
            }
            for row in counts
        ]
    
    def localize_box_counts_daily(self) -> dict:
        labeltypes: dict[int, FrameLabelType] = {
            flt.id: flt 
            for flt in self.db.session.query(FrameLabelType).all()
        }
        # TODO: future idea: videoIdsPassed = set() -> total videos, frames/video, boxes/video
        grouped_data = self.db.session.query(
            FrameLabel.labeldate,
            FrameLabel.labeltype,
            func.count().label("count")
        ).group_by(
            FrameLabel.labeldate, FrameLabel.labeltype
        ).order_by(
            FrameLabel.labeldate 
        ).all()

        # Zero counts
        def counts_per_type(labeltypes: dict[int, FrameLabelType]):
            return {id: 0 for id in labeltypes.keys()}
        
        # Iterate and to make daily counts for every item (even if they are absent)
        current_date = None
        daily_data = {}
        for row in grouped_data:
            rowdate = row.labeldate.strftime("%Y-%m-%d")
            if rowdate != current_date:
                daily_data[rowdate] = { 
                    'individual': counts_per_type(labeltypes), 
                    'cumulative': counts_per_type(labeltypes) if current_date is None else daily_data[current_date]['cumulative'].copy()
                }
                current_date = rowdate
            
            # Add count of current day, type to daily data
            daily_data[current_date]['individual'][row.labeltype] += row.count
            daily_data[current_date]['cumulative'][row.labeltype] += row.count

        return daily_data
    
    def localize_results(self) -> dict:
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
                    **recipe_results['results_dict']
                }
        return results
    
    def skills_prop_names(self, layercompositionname:str=None) -> list[str]:
        prop_name = case(
                (LayerComposition.name == None, LayerProperty.name),
                else_=LayerComposition.name
            ).label('prop_name')
        
        qry = self.db.session.query(
            LayerComposition,
            prop_name
        ).join(
            LayerComposition.property
        )

        if layercompositionname:
            qry = qry.filter(LayerComposition.compositionName == layercompositionname)

        return sorted({
            row.prop_name for row in qry.all()
        })
    
    def layercomposition_names(self) -> list[str]:
        layercomposition_names = self.db.session.query(
            LayerComposition.compositionName
        ).distinct(
            LayerComposition.compositionName
        ).all()
        return [lcn.compositionName for lcn in layercomposition_names]
    
    def skills_prop_counts(self, layercompositionname: str = None) -> dict:        
        prop_names = self.skills_prop_names(layercompositionname)

        query = self.db.session.query(
            *[
                func.sum(
                    func.round(
                        (
                            func.char_length(Skill.skillinfo) -
                            func.char_length(func.replace(Skill.skillinfo, prop_name, ""))
                        ) / func.char_length(prop_name)
                    )
                ).label(prop_name)
                for prop_name in prop_names
            ],
            self.split_train_test_skill
        ).group_by(
            self.split_train_test_skill
        )

        if layercompositionname:
            query = query.filter(
                func.json_search(Skill.skillinfo, 'one', layercompositionname) != None
            )

        results = query.all()

        output = {}
        for row in results:
            output[row.split]  = {prop_name: int(getattr(row, prop_name)) for prop_name in prop_names}

        return output

    def skills_prop_value_frequencies(self, layercompositionname: str = None) -> dict:
        """
        Returns counts of each distinct value for a given property in skillinfo JSON.
        Example: {'Backwards': {0: 12, 1: 59}, 'CrossRestriction': {71: 443, 72: 223, 73: 150...} }
        """
        query = self.db.session.query(
            func.json_extract(Skill.skillinfo, f'$.{layercompositionname}').label('skillinfo') if layercompositionname else Skill.skillinfo,
            self.split_train_test_skill
        ).group_by(
            func.json_extract(Skill.skillinfo, f'$.{layercompositionname}').label('skillinfo') if layercompositionname else Skill.skillinfo,
            self.split_train_test_skill
        )

        if layercompositionname:
            query = query.filter(
                func.json_search(Skill.skillinfo, 'one', layercompositionname) != None
            )

        result = query.all()

        prop_names = self.skills_prop_names(layercompositionname)
        counts = { prop_name: { s: Counter() for s in ['train', 'test'] } for prop_name in prop_names }
        for row in result:
            s = json.loads(row.skillinfo) if layercompositionname else row.skillinfo
            for k,v in extract_key_number_pairs(s):
                counts[k][row.split][v] += 1

        return {
            k: dict(values) for k, values in counts.items()
        }

    def skill_counts(self) -> dict:
        result = self.db.session.query(
            self.split_train_test_skill,
            func.count(self.split_train_test_skill).label('count')
        ).group_by(
            self.split_train_test_skill
        ).all()

        return { row.split: row.count for row in result }

    def skills_metrics(self) -> dict:
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

    def skill_counts_daily(self) -> dict:
        grouped_data = self.db.session.query(
            Skill.labeldate,
            self.split_train_test_skill,
            func.count().label("count")
        ).group_by(
            Skill.labeldate, self.split_train_test_skill
        ).order_by(
            Skill.labeldate
        ).all()

        # Zero counts
        def counts_per_split():
            return {'train': 0, 'test': 0}
        
        # Iterate and to make daily counts for every item (even if they are absent)
        current_date = None
        daily_data = {}
        for row in grouped_data:
            rowdate = row.labeldate.strftime("%Y-%m-%d")
            if rowdate != current_date:
                daily_data[rowdate] = { 
                    'individual': counts_per_split(), 
                    'cumulative': counts_per_split() if current_date is None else daily_data[current_date]['cumulative'].copy()
                }
                current_date = rowdate
            
            # Add count of current day, type to daily data
            daily_data[current_date]['individual'][row.split] += row.count
            daily_data[current_date]['cumulative'][row.split] += row.count

        return daily_data

