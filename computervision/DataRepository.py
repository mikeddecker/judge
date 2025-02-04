#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import sqlalchemy as sqlal
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
        print("rel paths of framelabels loaded")

    def __get_connection(self):
        HOST = '127.0.0.1'
        PORT = '3377'
        DATABASE = 'judge'
        USERNAME = 'root'
        PASSWORD = 'root'
        DATABASE_CONNECTION=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        engine = sqlal.create_engine(DATABASE_CONNECTION)#
        return engine.connect()

    def get_framelabels(self, train_test_val, type=1):
        # TODO : update with validation & 'random' sampling
        if train_test_val == "train":
            qry = sqlal.text(f"""SELECT * FROM FrameLabels WHERE MOD(videoId, 10) <> 5 AND labeltype = {type}""")

        if train_test_val == "val":
            qry = sqlal.text(f"""SELECT * FROM FrameLabels WHERE MOD(videoId, 10) = 5 AND labeltype = {type}""")

        if train_test_val == "test":
            raise ValueError(f"Changed test to val !!")
        return pd.read_sql(qry, con=self.con)

    def __load_relativePaths_of_videos_with_framelabels(self):
        relative_paths = {}
        qry = sqlal.text(f"""SELECT DISTINCT folderId, id, name FROM Videos WHERE id IN (SELECT DISTINCT videoId FROM FrameLabels)""")

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
        
        # df_videos["name"] = STORAGE_DIR + os.sep + df_videos["name"]
        df_videos.index = df_videos.id
        self.VideoNames = df_videos

    def save_train_results(self, df_history: pd.DataFrame, from_scratch: bool):
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

    def get_last_epoch_nr(self, modelname):
        """Return last epoch nr or 0"""
        qry = sqlal.text(f"""SELECT MAX(epoch) as last_epoch FROM TrainResults WHERE modelname = \'{modelname}\'""")
        df = pd.read_sql(qry, con=self.con)
        epoch = df.loc[0, 'last_epoch']
        epoch = 0 if epoch is None else epoch
        return epoch
    
    def get_last_epoch_values(self, modelname, epoch):
        qry = sqlal.text(f"""SELECT * FROM TrainResults WHERE modelname = \'{modelname}\' AND epoch = {epoch}""")
        return pd.read_sql(qry, con=self.con)
