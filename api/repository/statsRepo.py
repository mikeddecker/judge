import os
import pandas as pd
from config import RECIPES
from collections import Counter, defaultdict
from config import RECIPES, ENVS
from flask_sqlalchemy import SQLAlchemy
from helpers.helpers import load_json_file
from repository.models import Video as VideoInfoDB, FrameLabel, FrameLabelType
from repository.models import Skill, LayerComposition, Layer, LayerValue
from sqlalchemy import func, case
from sqlalchemy.orm import aliased
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
                    **recipe_results['results_dict'],
                    'ious': { **ious_all },
                }
        return results

    def get_localize_labelinfo_per_video(self):
        # Get videos with labels, including density and labeled frame numbers
        query = self.db.session.query(
            FrameLabel.videoId,
            FrameLabel.frameNr,
            func.count().label('boxes'),
            VideoInfoDB.name,
            VideoInfoDB.fps,
            VideoInfoDB.duration,
            VideoInfoDB.frameLength,
            VideoInfoDB.training
        ).join(VideoInfoDB, FrameLabel.videoId == VideoInfoDB.id).group_by(
            FrameLabel.videoId, FrameLabel.frameNr,
            VideoInfoDB.name, VideoInfoDB.fps, VideoInfoDB.duration, VideoInfoDB.frameLength, VideoInfoDB.training
        ).order_by(FrameLabel.videoId, FrameLabel.frameNr)

        results = query.all()

        video_dict = defaultdict(lambda: {'name': '', 'fps': 0, 'frameLength': 0, 'training': False, 'duration': 0, 'frames': []})

        for row in results:
            vid = row.videoId
            video_dict[vid]['frames'].append({'frameNr': row.frameNr, 'boxes': row.boxes})
            video_dict[vid]['name'] = row.name
            video_dict[vid]['fps'] = row.fps
            video_dict[vid]['frameLength'] = row.frameLength
            video_dict[vid]['training'] = row.training
            video_dict[vid]['duration'] = row.duration

        videos = []
        for vid, data in video_dict.items():
            labeled_frames = data['frames']
            total_boxes = sum(f['boxes'] for f in labeled_frames)
            labeled_frame_nrs = [f['frameNr'] for f in labeled_frames]
            density = len(labeled_frame_nrs) * data['fps'] / data['frameLength']
            videos.append({
                'id': vid,
                'name': data['name'],
                'duration': data['duration'],
                'fps': data['fps'],
                'frameLength': data['frameLength'],
                'frameCount': len(labeled_frame_nrs),
                'totalBoxes': total_boxes,
                'density': density,
                'labeledFrameNrs': sorted(labeled_frame_nrs)
            })

        return videos

    def skills_layerNames(self, layercompositionname:str=None) -> list[str]:
        qry = self.db.session.query(
            LayerComposition,
            Layer.name.label("layerName")
        ).join(
            LayerComposition
        )

        if layercompositionname:
            qry = qry.filter(LayerComposition.compositionName == layercompositionname)

        return sorted({
            row.layerName for row in qry.all()
        })

    def skill_prop_value_names_dataframe(self):
        query = self.db.session.query(
            Layer.id.label("layerId"),
            Layer.name.label("layerName"),
            Layer.type,
            Layer.min,
            Layer.max,
            Layer.step,
            LayerValue.id.label("value_id"),
            LayerValue.name.label("value_name"),
        ).outerjoin(
            LayerValue, Layer.id == LayerValue.layerId
        )
        return pd.read_sql(query.statement, self.db.engine)

    def layercomposition_names(self) -> list[str]:
        layercomposition_names = self.db.session.query(
            LayerComposition.compositionName
        ).distinct(
            LayerComposition.compositionName
        ).all()
        return [lcn.compositionName for lcn in layercomposition_names]

    def layer_counts(self, layercompositionname: str = None) -> dict:
        layerNames = self.skills_layerNames(layercompositionname)

        query = self.db.session.query(
            *[
                func.sum(
                    func.round(
                        (
                            func.char_length(Skill.skillinfo) -
                            func.char_length(func.replace(Skill.skillinfo, layerName, ""))
                        ) / func.char_length(layerName)
                    )
                ).label(layerName)
                for layerName in layerNames
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
            output[row.split]  = {layerName: int(getattr(row, layerName)) for layerName in layerNames}

        return output

    def layer_value_counts(self) -> dict:
        """
        Returns counts of each distinct value for a given layer in skillinfo JSON.
        Example: {'Backwards': {0: 12, 1: 59}, 'CrossRestriction': {71: 443, 72: 223, 73: 150...} }
        """
        df_prop_value_names = self.skill_prop_value_names_dataframe()
        def map_prop_value(df: pd.DataFrame, layerName: str, prop_value: str):
            df_layerName_filtered = df[df['layerName'] == layerName]
            if df_layerName_filtered.iloc[0]['type'] == 'categorical':
                if prop_value == 0:
                    return 0
                return df_layerName_filtered[df_layerName_filtered['value_id'] == prop_value].iloc[0]['value_name']
            else:
                return prop_value

        query = self.db.session.query(
            Skill.skillinfo,
            self.split_train_test_skill
        )

        result = query.all()

        layer_composition_names = self.layercomposition_names()
        layer_composition_names.append('total')
        counts = { lcn: { layerName: { s: Counter() for s in ['train', 'test'] } for layerName in self.skills_layerNames(None if lcn == 'total' else lcn) } for lcn in layer_composition_names }
        for row in result:
            for lcn in layer_composition_names:
                if lcn in row.skillinfo.keys() or lcn == 'total':
                    skill: dict = row.skillinfo if lcn == 'total' else row.skillinfo[lcn]
                    for layer, value in extract_key_number_pairs(skill):
                        counts[lcn][layer][row.split][value] += 1

        return {
            lcn: {
                layerName: {
                    split: {
                        map_prop_value(df_prop_value_names, layerName, value): count for value, count in value_counter.items() }
                    for split, value_counter in layerName_split_values.items() }
                for layerName, layerName_split_values in lcn_values.items() }
            for lcn, lcn_values in counts.items()
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
        return RECIPES
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
            Skill.createdAt,
            self.split_train_test_skill,
            func.count().label("count")
        ).group_by(
            Skill.createdAt, self.split_train_test_skill
        ).order_by(
            Skill.createdAt
        ).all()

        # Zero counts
        def counts_per_split():
            return {'train': 0, 'test': 0}

        # Iterate and to make daily counts for every item (even if they are absent)
        current_date = None
        daily_data = {}
        for row in grouped_data:
            rowdate = row.createdAt.strftime("%Y-%m-%d")
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

    def layer_composition_counts(self) -> dict:
        layer_composition_names = self.layercomposition_names()
        query = self.db.session.query(
            *[func.json_length(func.json_extract(Skill.skillinfo, f'$.{lcn}')).label(lcn) for lcn in layer_composition_names],
            func.count().label('count'),
            self.split_train_test_skill
        ).group_by(
            *[lcn for lcn in layer_composition_names],
            self.split_train_test_skill
        )

        results = query.all()

        dataset = defaultdict(lambda: Counter({'train': 0, 'test': 0}))
        for row in results:
            key_parts = [
                f"{row[index]} {lcn}"
                for index, lcn in enumerate(layer_composition_names)
                if row[index] not in (None, 0)
            ]
            key = ', '.join(key_parts) if key_parts else "empty"

            count = row.count
            split = row.split

            dataset[key][split] = count

        return {k: dict(v) for k, v in dataset.items()}

