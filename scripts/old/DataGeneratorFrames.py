import numpy as np
import pandas as pd
import keras
import cv2
import sqlalchemy as sqlal
from pymysql import OperationalError
from utils_cv2 import get_frames
from DataRepository import DataRepository
from keras.utils import to_categorical

class DataGeneratorRectangles(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train=True, dim=(128, 128), n_channels=3, batch_size=32,
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

        self.repo = DataRepository()

        print('DataGeneratorSkillBorders init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.batch_order)

    def __getitem__(self, batch_nr):
        'Generate one batch of data'
        # Generate batch df view
        print(f" __getitem__({batch_nr})")
        video_id = self.batch_order.iloc[batch_nr]['videoID']
        video_batch_nr = self.batch_order.iloc[batch_nr]['batch_nr_video']

        # print(f"get_rects: {video_id}", video_batch_nr, self.batch_size)
        df_labels = self.repo.get_rectangles_from_batch(videoID=video_id, batch_nr=video_batch_nr, batch_size=self.batch_size)
        if (len(df_labels) < self.batch_size):
            pass
            # print('df_labels: ', df_labels)
            # df_labels = self.fill_time_length_dimension(df_labels)

        y = np.array(df_labels[['rect_center_x', 'rect_center_y', 'rect_size']])

        # y = np.expand_dims(y, axis=0)
        # y = np.expand_dims(y, axis=-1)
         
        min_frame = df_labels.iloc[0]['frameNr']
        max_frame = df_labels.iloc[-1]['frameNr']
        path = '../' + self.repo.get_path(video_id)
        X = get_frames(path, min_frame, max_frame, dim=self.dim)

        # X = np.expand_dims(X, axis=0)  # Add batch dimension
        
        return X, y

    def fill_time_length_dimension(self, df_labels):
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