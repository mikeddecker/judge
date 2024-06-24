import numpy as np
import pandas as pd
import keras
import cv2
import sqlalchemy as sqlal
from pymysql import OperationalError
from utils_cv2 import get_frames
from DataRepository import DataRepository
from keras.utils import to_categorical

class DataGeneratorSkillBorders(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, train=True, batch_size=32, dim=(128, 128), n_channels=3,
                 n_classes=10, shuffle=True, time_length=16, axis=0, **kwargs):
        'Initialization'
        super().__init__(**kwargs)
        self.dim = dim
        self.train = train
        self.time_length = time_length
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
        if self.len is None:
            self.len = self.repo.get_randomized_borderlabels_and_batches_per_video(batch_size=self.batch_size, training=self.train).iloc[-1]['total_batches']
        print('len called', 'len is:', self.len)
        return self.len

    def __getitem__(self, batch_nr):
        'Generate one batch of data'
        # Generate batch df view
        df = self.batch_order
        print(' getitem, batch_nr', batch_nr, 'train is:', self.train)
        index = len(df[df['total_batches']<=batch_nr])
        video_id = df.iloc[index]['videoID']
        video_batch_nr = batch_nr - (df.iloc[index-1]['total_batches'] if index > 0 else 0)

        # print(batch_nr, ' vidID', video_id, '& vid_batch_nr', video_batch_nr)

        df_labels = self.repo.get_borderlabels_batch(videoID=video_id, batch_nr=video_batch_nr, batch_size=self.batch_size)
        if (len(df_labels) < 20):
            df_labels = self.fill_batch_size_dimension(df_labels)

        y = np.array(df_labels['label'])
        y = np.where(y == 9, 3, y)
        y = np.where(y == 8, 4, y)
        y = np.expand_dims(y, axis=0)
        y = np.expand_dims(y, axis=-1)
        
        min_frame = df_labels.iloc[0]['frameNr']
        max_frame = min_frame + self.batch_size - 1
        path = '../' + self.repo.get_path(video_id)
        X = get_frames(path, min_frame, max_frame, dim=self.dim)

        X = np.expand_dims(X, axis=0)  # Add batch dimension
        
        return X, y

    def fill_batch_size_dimension(self, df_labels):
        print('fill_batch_size_dimension_called')
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
        self.batch_order = self.repo.get_randomized_borderlabels_and_batches_per_video(self.batch_size, self.train)
        print(self.batch_order)

    def load_frame(self, path, frame_nr, dx, dy):
        print('load_frame_called')
        cap = cv2.VideoCapture(self.videofolder + path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
        res, frame = cap.read()
        if not res:
            raise ValueError(f"Failed to read frame {frame_nr} from {self.videofolder + path}")
        frame = cv2.resize(frame, (dx, dy))
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Uncomment if necessary
        cap.release()
        return frame / 255.0