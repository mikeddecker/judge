#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import sqlalchemy as sqlal
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

class DataRepository:
    VideoNames = {} # pandas dataframe

    def __init__(self):  
        self.con = self.__get_connection()
        print('connection established, loading frame paths')
        self.__load_relativePaths_of_videos_with_framelabels()
        print("relative paths of framelabels loaded")

    def __get_connection(self):
        HOST = '127.0.0.1'
        PORT = '3377'
        DATABASE = 'judge'
        USERNAME = 'root'
        PASSWORD = 'root'
        DATABASE_CONNECTION=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        engine = sqlal.create_engine(DATABASE_CONNECTION)#
        return engine.connect()

    def check_connection(connection):
        try:
            # Execute a simple query to check the connection
            connection.execute("SELECT 1")
            print("Connection is still alive.")
            return True
        except SQLAlchemyError:
            print("Connection lost.")
            return False


    def get_dd3_videoIds(self, ):
        # TODO : update with validation & 'random' sampling
        qry = sqlal.text(f"""SELECT id FROM Videos WHERE folderId = 3""")

        return pd.read_sql(qry, con=self.con)

    def get_framelabels(self, train_test_val, type=1):
        # TODO : update with validation & 'random' sampling
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM FrameLabels WHERE MOD(videoId, 10) <> 5 AND labeltype = {type}""")

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM FrameLabels WHERE MOD(videoId, 10) = 5 AND labeltype = {type}""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        return pd.read_sql(qry, con=self.con)
    
    def get_fully_segmented_videos(self, train_test_val, type='DD'):
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM Videos WHERE MOD(id, 10) <> 5 AND completed_skill_labels = 1""")  

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM Videos WHERE MOD(id, 10) = 5 AND completed_skill_labels = 1""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        return pd.read_sql(qry, con=self.con)

    def get_skills_of_fully_segmented_videos(self, train_test_val, type='DD'):
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) <> 5 AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)""")  

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) = 5 AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        return pd.read_sql(qry, con=self.con)

    def get_skills(self, train_test_val, type='DD', videoId:int=None):
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) <> 5""") # TODO segmentation:  AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)

        and_where_videoId = f"AND videoId = {videoId}" if videoId else ""
        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM Skillinfo_DoubleDutch WHERE MOD(videoId, 10) = 5 {and_where_videoId}""") # TODO segmentation:  AND videoId in (SELECT id FROM Videos WHERE completed_skill_labels = 1)

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        return pd.read_sql(qry, con=self.con)
    
    def get_skill_category_counts(self):
        qry = sqlal.text(f"""SELECT 
            (SELECT COUNT(*) FROM Skillinfo_DoubleDutch_Skill) AS skills,
            (SELECT COUNT(*) FROM Skillinfo_DoubleDutch_Type) AS types,
            (SELECT COUNT(*) FROM Skillinfo_DoubleDutch_Turner) AS turners;
        """)

        return pd.read_sql(qry, con=self.con)

    def __load_relativePaths_of_videos_with_framelabels(self):
        relative_paths = {}
        qry = sqlal.text(f"""SELECT DISTINCT folderId, id, name FROM Videos;""")

        df_videos = pd.read_sql(qry, con=self.con)
        for idx, row in df_videos.iterrows():
            folderId = int(row["folderId"])
            name = row["name"]
            childId = folderId
            subfolders = []
            while childId is not None:
                qry = sqlal.text(f"""SELECT parentId, name FROM Folders WHERE id = {childId}""")
                df_child = pd.read_sql(qry, self.con).iloc[0]
                subfolders.insert(0, df_child["name"])
                childId = df_child["parentId"]
        
            relative_paths[folderId] = os.path.join(*subfolders)
            df_videos.loc[idx,"name"] = os.path.join(*subfolders, name)
        
        df_videos.index = df_videos.id
        self.VideoNames = df_videos
        self.VideoNames.index = df_videos["id"]

    def save_train_results(self, df_history: pd.DataFrame, from_scratch: bool, skills: bool = False):
        if skills:
            return self.__save_train_results_skills(df_history, from_scratch)
        
        if from_scratch:
            delete_old = sqlal.text(f"""
                DELETE FROM TrainResults WHERE modelname = \'{df_history.loc[0,'modelname']}\'
            """)
            self.con.execute(delete_old)
            self.con.commit()

        insert = sqlal.text("""
            INSERT INTO TrainResults (modelname, train_date, epoch, iou, loss, val_iou, val_loss)
            VALUES (:modelname, :train_date, :epoch, :iou, :loss, :val_iou, :val_loss)
        """)

        for _, row in df_history.iterrows():
            self.con.execute(insert, {
                'modelname': row['modelname'],
                'train_date': row['train_date'],
                'epoch': row['epoch'],
                'iou': row['iou'],
                'loss': row['loss'],
                'val_iou': row['val_iou'],
                'val_loss': row['val_loss']
            })
        self.con.commit()
    
    def __save_train_results_skills(self, df_history: pd.DataFrame, from_scratch: bool):
        if from_scratch:
            delete_old = sqlal.text(f"""
                DELETE FROM TrainResultsSkills WHERE modelname = \'{df_history.loc[0,'modelname']}\'
            """)
            self.con.execute(delete_old)
            self.con.commit()

        insert = sqlal.text("""
            INSERT INTO TrainResultsSkills (modelname, train_date, epoch, loss, accuracy, val_loss, val_accuracy, losses_and_metrics)
            VALUES (:modelname, :train_date, :epoch, :loss, :accuracy, :val_loss, :val_accuracy, :losses_and_metrics)
        """)

        for _, row in df_history.iterrows():
            self.con.execute(insert, {
                'modelname': row['modelname'],
                'train_date': row['train_date'],
                'epoch': row['epoch'],
                'loss': row['loss'],
                'accuracy': row['accuracy'],
                'val_loss': row['val_loss'],
                'val_accuracy': row['val_accuracy'],
                'losses_and_metrics': row.to_json(),
            })
        self.con.commit()

    def get_last_epoch_nr(self, modelname, type=None):
        """Return last epoch nr or 0"""
        tablename = 'TrainResultsSkills' if type == 'DD' else 'TrainResults'
        qry = sqlal.text(f"""SELECT MAX(epoch) as last_epoch FROM {tablename} WHERE modelname = \'{modelname}\'""")
        df = pd.read_sql(qry, con=self.con)
        epoch = df.loc[0, 'last_epoch']
        epoch = 0 if epoch is None else epoch
        return epoch
    
    def get_last_epoch_values(self, modelname, epoch, type=None):
        tablename = 'TrainResultsSkills' if type == 'DD' else 'TrainResults'
        qry = sqlal.text(f"""SELECT * FROM {tablename} WHERE modelname = \'{modelname}\' AND epoch = {epoch}""")
        return pd.read_sql(qry, con=self.con)
