#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip3 install sqlalchemy PyMySQL')


# In[2]:


import pandas as pd
import numpy as np
import sqlalchemy as sqlal
from pymysql import OperationalError

from utils_misc import pickle_load_or_create, pickle_save
from utils_cv2 import get_frame, show_frame_and_await_key, label_frames_from_df


# In[3]:


class DataRepository:
    def __init__(self):  
        self.con = self.get_connection()
        print('connection established')

    def get_connection(self):
        HOST = '127.0.0.1'
        PORT = '3377'
        DATABASE = 'judge'
        USERNAME = 'root'
        PASSWORD = 'root'
        DATABASE_CONNECTION=f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
        engine = sqlal.create_engine(DATABASE_CONNECTION)#
        return engine.connect()

    def executeSqlScriptsFromFile(self, filename):
        fd = open(filename, 'r')
        sqlFile = fd.read()
        fd.close()
        
        sqlCommands = sqlFile.split(';')
    
        for command in sqlCommands:
            try:
                self.con.execute(sqlal.text(command))
                self.con.commit()
            except OperationalError as msg:
                print("Command skipped: ", msg)

    def initDatabase(self):
        try:
            self.executeSqlScriptsFromFile('../db/init.sql')
        except Exception as e:
            print(e)

    def insert_folders(self):
        qry = "INSERT INTO Folders (path) VALUES ('videos'), ('belgium/bk'), ('belgium/pk')"
        self.con.execute(sqlal.text(qry))
        self.con.commit()

    def add_video(self, folder_id, name, training, obstruction, quality, type, dimensionX, dimensionY):
        command = sqlal.text(f"""
        INSERT INTO Videos (folderID, name, training, obstruction, quality, type, dimensionX, dimensionY) 
        VALUES ({folder_id}, \'{name}\', {training}, {obstruction}, {quality}, {type}, {dimensionX}, {dimensionY})
        """)
        self.con.execute(command)
        self.con.commit()

    
    def insert_videos(self):
        video_names_train = [
            '20240201_atelier_001.mp4',
            '20240201_atelier_002.mp4',
            '20240201_atelier_003.mp4',
            '20240201_atelier_004.mp4',
            '20240201_atelier_005.mp4',
            '20240209_atelier_006.mp4',
            '20240209_atelier_007.mp4',
            '20240209_atelier_008.mp4',
        ]
        
        video_names_test = [
            '20240209_atelier_009.mp4',
        ]
    
        for vid in video_names_train:
            self.add_video(1, vid, 1, 0, 70, 2, 1920, 1080)
        for vid in video_names_test:
            self.add_video(1, vid, 0, 0, 70, 2, 1920, 1080)
    
    def get_video_name_id_dict(self):    
        qry = sqlal.text("SELECT name, videoID FROM Videos")
        dict_videos = {}
        for idx, row in pd.read_sql_query(qry, self.con).iterrows():
            dict_videos[row['name']] = row['videoID']
        return dict_videos

    def add_borderlabel(self, videoID, frameNR, label, manual=1):    
        command = sqlal.text(f"""
        INSERT INTO FrameLabels values ({videoID}, {frameNR}, {label}, {manual})
        """)
        self.con.execute(command)
        self.con.commit()
    
    def insert_framelabels(self):
        df_labels = pickle_load_or_create('df_video_border_labels_seq', lambda: pd.DataFrame())
        video_dict = self.get_video_name_id_dict()
        for _, row in df_labels.iterrows():
            vid_id = video_dict[row['path']]
            self.add_borderlabel(vid_id, row['frame'], row['border'])
    
    def refreshDatabase(self):
        print('start init')
        self.initDatabase()
        print('init done')
        self.insert_folders()
        print('insert folders done')
        self.insert_videos()
        print('insert videos done')
        self.insert_framelabels()
        print('insert labels done')
    
    def label_exists(self, vid_id, frame_nr, label):
        qry = f"SELECT COUNT(*) as count FROM FrameLabels WHERE videoID = {vid_id} AND frameNr = {frame_nr} AND label = {label};"
        return (pd.read_sql_query(qry, self.con) > 0)['count'][0]
    
    def query_framelabels(self):
        qry=sqlal.text(f"""SELECT DISTINCT * FROM FrameLabels""")
        return pd.read_sql_query(qry, self.con)
    
    def get_randomized_borderlabels_and_batches_per_video(self, batch_size=16, training=True):
        qry = f"""
        WITH batches_per_video AS (
            SELECT f.videoID, CEILING(max(frameNr) / {batch_size}) as batches_in_video,
            ROW_NUMBER() OVER (ORDER BY RAND()) AS random_order
            FROM FrameLabels f
            JOIN Videos v ON f.videoID = v.videoID
            WHERE v.training = {training}
            GROUP BY f.videoID 
        )
        SELECT *,
        SUM(batches_in_video) OVER (ORDER BY random_order ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) as total_batches
        FROM batches_per_video ORDER BY random_order
        """
        df = pd.read_sql_query(qry, self.con)
        df['total_batches'] = df.total_batches.astype(int)
    
        return df
        
    def get_borderlabels_batch(self, videoID, batch_nr, batch_size=16):    
        min = batch_size * batch_nr
        max = min + batch_size - 1
        qry = f"""
        SELECT frameNr, label 
        FROM FrameLabels 
        WHERE videoID = {videoID} AND frameNr BETWEEN {min} AND {max}
        """
        df = pd.read_sql_query(qry, self.con)

        return df

    def get_path(self, videoID=1):
        qry = f"""
        SELECT CONCAT(f.path, \'/\', v.name) as path
        FROM Videos v
        JOIN Folders f
        ON v.folderID = f.folderID
        WHERE videoID = {videoID}
        """
        df = pd.read_sql_query(qry, self.con)['path'][0]
        return df


