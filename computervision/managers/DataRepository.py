#!/usr/bin/env python
# coding: utf-8
import json
import numpy as np
import os
import pandas as pd
import sqlalchemy as sqlal

from collections import defaultdict
from constants import ENVS
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError

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

class DataRepository:
    VideoNames = {} # pandas dataframe

    def __init__(self):  
        HOST = ENVS.DATABASE.MYSQLDB_HOST
        PORT = ENVS.DATABASE.MYSQLDB_LOCAL_PORT
        DATABASE = ENVS.DATABASE.MYSQLDB_DATABASE
        USERNAME = ENVS.DATABASE.MYSQLDB_USERNAME
        PASSWORD = ENVS.DATABASE.MYSQLDB_ROOT_PASSWORD
        DATABASE_CONNECTION=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        self.engine = sqlal.create_engine(DATABASE_CONNECTION, pool_recycle=30)#
        self.__load_relativePaths_of_videos_with_framelabels()

    def __get_connection(self):
        print(self.engine.pool.status())
        return self.engine.connect()
    
    def get_videoinfo(self, videoId):
        qry = sqlal.text(f"""SELECT * FROM Videos WHERE id = {videoId}""")
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)
        
    def get_framelabels(self, train_test_val):
        # TODO : update with validation & 'random' sampling
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM FrameLabels WHERE MOD(videoId, 10) <> 5 ORDER BY videoId, frameNr""")

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM FrameLabels WHERE MOD(videoId, 10) = 5 ORDER BY videoId, frameNr""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)
            
    def get_unique_videoId_frameNr(self, train_test_val):
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT videoId, frameNr FROM FrameLabels WHERE MOD(videoId, 10) <> 5 GROUP BY videoId, frameNr ORDER BY videoId, frameNr""")

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT videoId, frameNr FROM FrameLabels WHERE MOD(videoId, 10) = 5 GROUP BY videoId, frameNr ORDER BY videoId, frameNr""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)
        
    def get_fully_segmented_videos(self, train_test_val, type='DD'):
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM Videos WHERE MOD(id, 10) <> 5 AND completed_skill_labels = 1""")  

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM Videos WHERE MOD(id, 10) = 5 AND completed_skill_labels = 1""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)
        
    def get_recognition_config(self):
        qry_layers = f"""
            SELECT lp.id as propertyId, lp.name as layer, lp.type, lpv.name as value, lpv.id as valueId, lp.min, lp.max, lp.step
            FROM LayerProperties lp
            LEFT OUTER JOIN LayerPropertyValues lpv ON lp.id = lpv.propertyId
            ORDER BY lp.id, lpv.name;
        """
        with self.__get_connection() as connection:
            df_layers = pd.read_sql(qry_layers, con=connection)
            
            qry_compositions = f"""
                SELECT lc.compositionName, lc.stage, lc.propertyId,
                CASE
                    WHEN lc.name is NULL THEN lp.name
                    ELSE lc.name
                END AS name
                FROM LayerComposition lc
                JOIN LayerProperties lp ON lp.id = lc.propertyId
                ORDER BY lc.compositionName, lc.stage, lc.propertyId
            """
            df_composition = pd.read_sql(qry_compositions, con=connection)

            distinct_compositions = df_composition['compositionName'].unique()

            json_qry= ', '.join([f"MAX(JSON_LENGTH(skillinfo->'$.{cn}')) AS {cn}" for cn in distinct_compositions])

            df_max_lengths = pd.read_sql(f"""SELECT {json_qry} FROM Skills;""", con=connection).iloc[0]

        return df_layers, df_composition, df_max_lengths
            
    def get_team_boxes(self) -> pd.DataFrame:
        qry = sqlal.text("""
            SELECT
                videoId,
                frameNr,
                MIN(x - width / 2) AS xmin,
                MAX(x + width / 2) AS xmax,
                MIN(y - height / 2) AS ymin,
                MAX(y + height / 2) AS ymax,
                MAX(x + width / 2) - MIN(x - width / 2) AS width,
                (MAX(x + width / 2) + MIN(x - width / 2)) / 2 AS x,
                MAX(y + height / 2) - MIN(y - height / 2) AS height,
                (MAX(y + height / 2) + MIN(y - height / 2)) / 2 AS y    
            FROM FrameLabels
            WHERE MOD(videoId, 10) = 5 AND labeltype = 1
            GROUP BY videoId, frameNr
            ORDER BY videoId, frameNr
        """)
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)

    def get_videoIds_of_videos_with_skills(self) -> list[int]:
        with self.__get_connection() as connection:
            qry = sqlal.text(f"""SELECT DISTINCT videoId FROM Skills""")  
            return pd.read_sql(qry, con=connection)['videoId'].to_list()

    def get_skills_of_fully_segmented_videos(self, train_test_val):
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM Skills WHERE MOD(videoId, 10) <> 5 AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)""")  

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM Skills WHERE MOD(videoId, 10) = 5 AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)

    def get_skills(self, train_test_val, videoId:int=None):
        """videoId is optional, then it returns only skills from that videoId"""
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM Skills WHERE MOD(videoId, 10) <> 5""") # TODO segmentation:  AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)

        and_where_videoId = f"AND videoId = {videoId}" if videoId else ""
        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM Skills WHERE MOD(videoId, 10) = 5 {and_where_videoId}""") # TODO segmentation:  AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        
        with self.__get_connection() as connection:
            df = pd.read_sql(qry, con=connection)
            # Convert 'skillinfo' column from JSON string to Python dict
            if 'skillinfo' in df.columns:
                df['skillinfo'] = df['skillinfo'].apply(json.loads)
            return df
                
    def __load_relativePaths_of_videos_with_framelabels(self):
        with self.__get_connection() as connection:
            relative_paths = {}
            qry = sqlal.text(f"""SELECT DISTINCT folderId, id, name FROM Videos;""")

            df_videos = pd.read_sql(qry, con=connection)
            for idx, row in df_videos.iterrows():
                folderId = int(row["folderId"])
                name = row["name"]
                childId = folderId
                subfolders = []
                while childId is not None:
                    qry = sqlal.text(f"""SELECT parentId, name FROM Folders WHERE id = {childId}""")
                    df_child = pd.read_sql(qry, connection).iloc[0]
                    subfolders.insert(0, df_child["name"])
                    childId = df_child["parentId"]
            
                relative_paths[folderId] = os.path.join(*subfolders)
                df_videos.loc[idx,"name"] = os.path.join(*subfolders, name)
            
            df_videos.index = df_videos.id
            self.VideoNames = df_videos
            self.VideoNames.index = df_videos["id"]

    def save_train_results(self, df_history: pd.DataFrame, from_scratch: bool, skills: bool = False):
        with self.__get_connection() as connection:
            if skills:
                return self.__save_train_results_skills(df_history, from_scratch)
            
            if from_scratch:
                delete_old = sqlal.text(f"""
                    DELETE FROM TrainResults WHERE modelname = \'{df_history.loc[0,'modelname']}\'
                """)
                connection.execute(delete_old)
                connection.commit()

            insert = sqlal.text("""
                INSERT INTO TrainResults (modelname, train_date, epoch, iou, loss, val_iou, val_loss)
                VALUES (:modelname, :train_date, :epoch, :iou, :loss, :val_iou, :val_loss)
            """)

            for _, row in df_history.iterrows():
                connection.execute(insert, {
                    'modelname': row['modelname'],
                    'train_date': row['train_date'],
                    'epoch': row['epoch'],
                    'iou': row['iou'],
                    'loss': row['loss'],
                    'val_iou': row['val_iou'],
                    'val_loss': row['val_loss']
                })
            connection.commit()
    
    def __save_train_results_skills(self, df_history: pd.DataFrame, from_scratch: bool):
        with self.__get_connection() as connection:
            if from_scratch:
                delete_old = sqlal.text(f"""
                    DELETE FROM TrainResultsSkills WHERE modelname = \'{df_history.loc[0,'modelname']}\'
                """)
                connection.execute(delete_old)
                connection.commit()

            insert = sqlal.text("""
                INSERT INTO TrainResultsSkills (modelname, train_date, epoch, loss, accuracy, val_loss, val_accuracy, losses_and_metrics)
                VALUES (:modelname, :train_date, :epoch, :loss, :accuracy, :val_loss, :val_accuracy, :losses_and_metrics)
            """)

            for _, row in df_history.iterrows():
                connection.execute(insert, {
                    'modelname': row['modelname'],
                    'train_date': row['train_date'],
                    'epoch': row['epoch'],
                    'loss': row['loss'],
                    'accuracy': row['accuracy'],
                    'val_loss': row['val_loss'],
                    'val_accuracy': row['val_accuracy'],
                    'losses_and_metrics': row.to_json(),
                })
            connection.commit()

    def get_last_epoch_nr(self, modelname, type=None):
        """Return last epoch nr or 0"""
        with self.__get_connection() as connection:
            tablename = 'TrainResultsSkills' if type == 'DD' else 'TrainResults'
            qry = sqlal.text(f"""SELECT MAX(epoch) as last_epoch FROM {tablename} WHERE modelname = \'{modelname}\'""")
            df = pd.read_sql(qry, con=connection)
            epoch = df.loc[0, 'last_epoch']
            epoch = 0 if epoch is None else epoch
            return epoch
    
    def get_last_epoch_values(self, modelname, epoch, type=None):
        tablename = 'TrainResultsSkills' if type == 'DD' else 'TrainResults'
        qry = sqlal.text(f"""SELECT * FROM {tablename} WHERE modelname = \'{modelname}\' AND epoch = {epoch}""")
        with self.__get_connection() as connection:
            return pd.read_sql(qry, con=connection)
            
    def get_next_job(self):
        with self.__get_connection() as connection:
            qry = sqlal.text(f"""SELECT * FROM Jobs""")
            df_jobs = pd.read_sql(qry, con=connection)
            return None if len(df_jobs) == 0 else df_jobs.iloc[0]
    
    def delete_job(self, jobId:int):
        print(f"Deleting job ({jobId})")
        with self.__get_connection() as connection:
            qry = sqlal.text(f"""DELETE FROM Jobs WHERE id = :id""")
            connection.execute(qry, {'id': jobId})
            connection.commit()

    def get_video_path(self, videoId):
        return os.path.join(ENVS.DIRS.VIDEOS, self.VideoNames.loc[videoId, "name"])
    
    def get_frame_label_types(self):
        with self.__get_connection() as connection:
            qry = sqlal.text(f"""SELECT info FROM FrameLabelTypes""")
            flts = pd.read_sql(qry, con=connection)
            return flts['info'].to_list()

    def get_skill_prop_counts(self):
        with self.__get_connection() as connection:
            distinct_prop_names = """
                SELECT
                DISTINCT CASE
                    WHEN lc.name is NULL THEN lp.name
                    ELSE lc.name
                END AS name
                FROM LayerComposition lc
                JOIN LayerProperties lp ON lp.id = lc.propertyId
            """
            distinct_prop_names = pd.read_sql(distinct_prop_names, con=connection)['name'].to_list()

            connection.execution_options(stream_results=True)

            # counts[prop_name][value] = occurrence count
            counts = defaultdict(lambda: defaultdict(int))
            for chunk_dataframe in pd.read_sql("SELECT skillinfo FROM Skills", connection, chunksize=50000):
                print(f"Dataframe with {len(chunk_dataframe)} rows")

                for row in chunk_dataframe.itertuples(index=False):
                    s = json.loads(row.skillinfo)
                    for k,v in extract_key_number_pairs(s):
                        counts[k][v] += 1

            return counts

