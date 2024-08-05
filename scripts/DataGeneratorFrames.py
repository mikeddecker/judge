import random
import numpy as np
import pandas as pd
import keras
import cv2
import sqlalchemy as sqlal
from pymysql import OperationalError
from utils_cv2 import get_squared_frames
from DataRepository import DataRepository
from keras.utils import to_categorical

class DataGeneratorRectangles(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, rootfolder, train=True, dim=(128, 128), n_channels=3, batch_size=16,
                 n_classes=3, shuffle=True, axis=0, **kwargs):
        'Initialization'
        super().__init__(**kwargs)
        self.dim = dim
        self.train = train
        self.batch_size = batch_size
        self.n_channels = n_channels  # RGB or gray
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.len = None
        self.axis=axis
        self.rootfolder=rootfolder
        self.previous_whole_division = None

        self.repo = DataRepository()

        print('DataGeneratorSkillBorders init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.batch_order)

    def __getitem__(self, batch_nr):
        'Generate one batch of data'
        
        whole_division = batch_nr // self.batch_size
        modulo = batch_nr % self.batch_size
        if whole_division != self.previous_whole_division:
            # Fetch next {batch_size} frames and labels
            items = []

            for i in np.arange(self.batch_size):
                # Create tuples (x, y) using zip
                X, y = self.get_batch(whole_division * self.batch_size + i)
                data_pairs = list(zip(X, y))
                
                for i, pair in enumerate(data_pairs):
                    items.append(pair)

            # shuffle
            random.shuffle(items)

            # divide in {batch_size}d groups
            # Extract shuffled X and y
            self.X_whole_division, self.y_whole_division = zip(*items)
            self.previous_whole_division = whole_division
            
        # return {X and y[modulo]}
        start_index = whole_division * self.batch_size
        row_start = self.batch_order.loc[start_index:start_index + modulo - 1, 'frames'].sum()
        row_end = self.batch_order.loc[start_index:start_index + modulo, 'frames'].sum()
        #print(row_start, row_end, len(self.X_whole_division), len(self.y_whole_division))
        return np.array(self.X_whole_division[row_start:row_end]), np.array(self.y_whole_division[row_start:row_end])
    
    def get_batch(self, batch_nr):
        video_id = self.batch_order.iloc[batch_nr]['videoID']
        frame_start = self.batch_order.iloc[batch_nr]['frame_start']
        frame_end = self.batch_order.iloc[batch_nr]['frame_end']

        df_labels = self.repo.get_rectangles_from_batch(videoID=video_id, frame_start=frame_start, frame_end=frame_end)

        y = np.array(df_labels[['rect_center_x', 'rect_center_y', 'rect_size']])

        path = self.rootfolder + self.repo.get_path(video_id)
        X = get_squared_frames(path, frame_start, frame_end, dim=self.dim)

        return X, y

    def fill_time_length_dimension(self, df_labels):
        # TODO
        print('fill_time_length_dimension_called')
        min_frame = df_labels.iloc[-1]['frameNr'] + 1
        max_frame_exlcuded = df_labels.iloc[0]['frameNr'] + self.batch_size
        arr = np.arange(min_frame, max_frame_exlcuded)
        df_fill = pd.DataFrame({
            'frameNr': [7777 for i in arr],
            'label': [0 for i in arr]
        })
        return pd.concat([df_labels, df_fill])

    def on_epoch_end(self):
        print('on_epoch_end_called')
        'Updates indexes after each epoch'
        # shuffle, insert indexes remain, position in df changes
        self.batch_order = self.repo.get_batch_order_frames(self.batch_size, self.train)
        # print(self.batch_order)