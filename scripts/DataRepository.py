#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip3 install sqlalchemy PyMySQL')


# In[2]:


import pandas as pd
import numpy as np
import sqlalchemy as sqlal
from pymysql import OperationalError
import os

from utils_misc import pickle_load_or_create, pickle_save
from utils_cv2 import get_frame, show_frame_and_await_key, label_frames_from_df, get_video_length


# In[3]:


class DataRepository:
    # TODO : Exceptions, on wrong query, rollback transaction
    
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

    def read_sql(self, qry):
        df = pd.read_sql_query(qry, self.con)
        self.con.commit()
        return df

    def execute(self, cmd):
        """Execute function, to make sure it's always commited."""
        self.con.execute(sqlal.text(cmd))
        self.con.commit()

    def initDatabase(self):
        try:
            self.executeSqlScriptsFromFile('../db/init.sql')
        except Exception as e:
            print(e)

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
        for idx, row in self.read_sql(qry).iterrows():
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
        return (self.read_sql(qry) > 0)['count'][0]
    
    def query_framelabels(self, videoID):
        qry=sqlal.text(f"""SELECT * FROM FrameLabels WHERE videoID = {videoID} ORDER BY FrameNr""")
        return self.read_sql(qry)
    
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
        df = self.read_sql(qry)
        df['total_batches'] = df.total_batches.astype(int)
    
        return df

    def get_batch_order_frames(self, batch_size=16, training=True):
        # TODO : Update procedure in workbench: case when deleted frames --> partition over rank - group by ...
        qry = f"""CALL GetFrameBatchNrs({batch_size}, {training})"""
        df = self.read_sql(qry)
        df['batch_nr_video'] = df.batch_nr_video.astype(int)
        df['batch_id'] = df.batch_id.astype(int)
    
        return df
    
    def get_rectangles_from_batch(self, videoID, batch_nr, batch_size=16):
        # TODO : Update procedure : videoID, frameStart, frameEnd is most simple
        qry = f"""
        CALL GetRectLabels({videoID}, {batch_nr}, {batch_size})
        """
        df = self.read_sql(qry)

        return df
    
    def get_borderlabels_batch(self, videoID, batch_nr, batch_size=16):    
        min = batch_size * batch_nr
        max = min + batch_size - 1
        qry = f"""
        SELECT frameNr, label 
        FROM FrameLabels 
        WHERE videoID = {videoID} AND frameNr BETWEEN {min} AND {max}
        """
        df = self.read_sql(qry)

        return df

    def get_path(self, videoID=1):
        qry = f"""
        SELECT CONCAT(f.path, \'/\', v.name) as path
        FROM Videos v
        JOIN Folders f
        ON v.folderID = f.folderID
        WHERE videoID = {videoID}
        """
        df = self.read_sql(qry)['path'][0]
        return df

    def execute_command(self, command):
        self.con.execute(command)
        self.con.commit()

    def fetch_qry(self, qry):
        return self.read_sql(qry)

    def add_border(self, videoID, frame_start, frame_end, manual_insert):
        if not self.is_valid_border(videoID, frame_start, frame_end):
            raise ValueError('Invalid frame skillborder')
        
        command = sqlal.text(f"""INSERT INTO Borders VALUE ({videoID}, {frame_start}, {frame_end}, {manual_insert});""")
        self.execute_command(command)

    def get_borders(self, videoID):
        qry = sqlal.text(f"""SELECT * FROM Borders WHERE videoID = {videoID}""")
        return self.fetch_qry(qry)

    def is_valid_border(self, videoID, start, end):
        MIN_SKILL_LENGTH = 3
        if end <= start:
            return False
        if end - start < MIN_SKILL_LENGTH:
            return False
        return 0 == len(self.get_border_overlap(videoID, start, end))
    
    def get_border_overlap(self, videoID, start, end):
        qry = sqlal.text(f"""
        SELECT *
        FROM Borders
        WHERE videoID = {videoID} AND 
        (({start} <= frame_start AND {end} > frame_start) OR ({end} >= frame_end AND {start} < frame_end))
        """)
        return self.fetch_qry(qry)

    def remove_border(self, videoID, start, end):
        command = sqlal.text(f"""DELETE FROM Borders WHERE videoID = {videoID} AND frame_start = {start} AND frame_end = {end}""")
        self.execute_command(command)

    def uninserted_borders_to_framelabels(self):
        qry = sqlal.text(f"""
        SELECT * FROM Borders
        WHERE videoID IN (SELECT videoID FROM Videos WHERE manually_bordered = 1 AND borderlabels_added = 0)
        """)
        df_borders = self.fetch_qry(qry)
        video_ids = df_borders.videoID.unique()
        print(video_ids)

        def frameNrs_for_column(df_borders, video_id, column, sign):
            return pd.concat([df_borders[df_borders.videoID == video_id][column], df_borders[df_borders.videoID == video_id][column] + sign]).sort_values().values
        
        for vid_id in video_ids:
            print('doing video', vid_id)
            path = '../' + self.get_path(vid_id)
            length = get_video_length(path)
            labels = np.array([0 for _ in np.arange(length)])
            
            start_indexes = frameNrs_for_column(df_borders, vid_id, 'frame_start', 1)
            end_indexes = frameNrs_for_column(df_borders, vid_id, 'frame_end', -1)
            skill_indexes = []
            for _, row in df_borders[df_borders.videoID == vid_id].iterrows():
                skill_indexes.extend(range(row['frame_start'] + 1, row['frame_end']))
        
            labels[skill_indexes] = 2
            labels[start_indexes] = 1
            labels[end_indexes] = 1
        
            values = ", ".join([f'({vid_id if vid_id > 0 else len(labels)}, {idx}, {lbl},1)' for idx, lbl in enumerate(labels)])

            command = sqlal.text(f"""INSERT INTO FrameLabels (videoID, frameNr, label, manual_insert) VALUES {values}""")
            self.execute_command(command)

            # Fixin 0 to be at the end. last frame should always be no skipping, just like the first frame.
            # Or that may be our bias :)
            command = sqlal.text(f"UPDATE FrameLabels SET frameNr = {len(labels)} WHERE videoID = {vid_id} AND frameNr = 0")
            self.execute_command(command)

            # Mark as labeled
            command = sqlal.text(f"UPDATE Videos SET borderlabels_added = 1 WHERE videoID = {vid_id}")
            self.execute_command(command)


    
    def update_rectangle(self, vid_id, frame_nr, rx, ry, rsize):
        command = sqlal.text(f"""
        UPDATE FrameLabels SET rect_center_x = {rx}, rect_center_y = {ry}, rect_size = {rsize}  
        WHERE videoID = {vid_id} AND frameNr = {frame_nr}""")
        self.execute_command(command)

# ------------------------------- 
#   Folders
# ------------------------------- 
    def exists_folder(self, foldername):
        qry = ""
        self.con.execute(sqlal.text(qry))
        self.con.commit()

    def add_folder(self, foldername):
        pass

    def get_folderID(self, foldername):
        pass

    def get_folder(self, folderID):
        pass

# -------------------------------
#    Club
# ------------------------------- 
    def exists_club(self, club):
        qry = ""
        self.execute(qry)

    def add_club(self, club):
        pass

    def get_clubID(self, club):
        pass
# -------------------------------
#    Competition
# ------------------------------- 
    def exists_competition(self, competition):
        """Does EK, BK, PK... already exists in DB?"""
        qry = ""
        self.execute(qry)

    def add_competition(self, competition):
        pass

    def get_competitionID(self, competition):
        pass

# -------------------------------
#    Discipline
# ------------------------------- 
    def exists_discipline(self, discipline):
        """Does SR, DD4, SR4, CW... already exists in DB?"""
        qry = ""
        self.execute(qry)

    def add_discipline(self, discipline):
        pass

    def get_disciplineID(self, discipline):
        pass


# -------------------------------
#    Age
# ------------------------------- 
    def exists_age(self, categorie):
        """Does junioren, senioren... already exists in DB?"""
        qry = ""
        self.execute(qry)

    def add_age(self, categorie):
        pass

    def get_ageID(self, categorie):
        pass


# -------------------------------
#    Video
# ------------------------------- 
    def exists_video(self, videoname):
        qry = ""
        self.execute(qry)

    def add_video(self, folder, videoname, competition, club, discipline, age):
        vid_length = 0

        # Add and/or get folderID + vid_length
        if isinstance(folder, str):
            vid_length = get_video_length(os.path.join(folder, videoname))
        
            if not self.exists_folder(folder):
                self.add_folder(folder)
            folder = self.get_folderID(folder)
        else:
            path = os.path.join(self.get_folder(folder).iloc[0]['path'], videoname)
            vid_length = get_video_length(path)

        # Add and/or get competitionID
        if isinstance(competition, str):
            if not self.exists_competition(competition):
                self.add_competition(competition)
            competition = self.get_competitionID(competition)
        
        # Add and/or get clubID
        if isinstance(club, str):
            if not self.exists_club(club):
                self.add_club(club)
            club = self.get_clubID(club)
        
        # Add and/or get ageID
        if isinstance(age, str):
            if not self.exists_age(age):
                self.add_age(age)
            age = self.get_ageID(age)

        # Add and/or get disciplineID
        if isinstance(discipline, str):
            if not self.exists_discipline(discipline):
                self.add_discipline(discipline)
            discipline = self.get_disciplineID(discipline)

    # TODO : qry = ... & excecute + testings


    def explore_and_add_videos(self):
        pass
