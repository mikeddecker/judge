from flask_sqlalchemy import SQLAlchemy
from repository.models import Video as VideoInfoDB, Folder as FolderDB, FrameLabel, Skillinfo_DoubleDutch, Skillinfo_DoubleDutch_Skill, Skillinfo_DoubleDutch_Turner, Skillinfo_DoubleDutch_Type, FrameLabelType
from sqlalchemy import desc, func
from typing import List

class StatsRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db

    def localize_framelabels_total(self):
        counts = self.db.session.query(
            FrameLabel.labeltype,
            func.count().label("count")
        ).group_by(
            FrameLabel.labeltype
        ).all()
        return [
            {
                "type": row.labeltype,
                "count": row.count
            }
            for row in counts
        ]
    
    def localize_framelabels_daily(self) -> dict:
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