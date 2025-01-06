#!/usr/bin/env python
# coding: utf-8
import os
import numpy as np
import pandas as pd
import sqlalchemy as sqlal

from dotenv import load_dotenv

load_dotenv()

STORAGE_DIR = os.getenv("STORAGE_DIR")

class DataRepository:
    VideoNames = {} # pandas dataframe

    def __init__(self):  
        self.con = self.__get_connection()
        print('connection established')

    def __get_connection(self):
        HOST = '127.0.0.1'
        PORT = '3377'
        DATABASE = 'judge'
        USERNAME = 'root'
        PASSWORD = 'root'
        DATABASE_CONNECTION=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        engine = sqlal.create_engine(DATABASE_CONNECTION)#
        return engine.connect()

    def get_framelabels(self, train_test_val):
        qry = sqlal.text(f"""SELECT * FROM FrameLabels""")
        return pd.read_sql(qry, con=self.con)

    def load_relativePaths_of_videos_with_framelabels(self):
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

