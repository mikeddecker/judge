import keras
import numpy as np
import pandas as pd
from DataRepository import DataRepository

# TODO : change to tf dataset, so prefetch is possible https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGeneratorFrames(keras.utils.Sequence):
    def __init__(self,
                 frameloader,
                 train_test_val: str, # train, test, val
                 dim: tuple, # e.g. (128,128)
                 batch_size=32, # Default batch size
                 normalized=True,
                 as_video=False,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dim, tuple)
        assert len(dim) == 2
        assert isinstance(train_test_val, str)
        assert train_test_val in ['train', 'test', 'val']
        self.dim = dim
        self.train_test_val = train_test_val
        self.augment = train_test_val == 'train'
        self.batch_size = batch_size
        self.frameloader = frameloader
        self.repo = DataRepository()
        self.Frames = self.repo.get_framelabels(train_test_val)
        self.as_video = as_video

        print('DataGeneratorSkillBorders init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.Frames) // self.batch_size

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        # print("getitem", batch_nr)
        frames = []
        i_start = batch_nr * self.batch_size
        i_end = min(len(self.Frames), (batch_nr + 1) * self.batch_size)
        y_values = []
        for i in range(i_start, i_end):
            row = self.Frames.iloc[i]
            videoId = row["videoId"]
            frameNr = row["frameNr"]
            x, y, w, h = self.Frames.iloc[i][["x", "y", "width", "height"]]
            try:
                if self.augment:
                    loaded_frame, y = self.frameloader.get_frame(videoId, frameNr, self.dim[0], x, y, w, h, printId=False)
                else:
                    loaded_frame, y = self.frameloader.get_frame_original(videoId, frameNr, self.dim[0], x, y, w, h, printId=False)
                y_values.append(y)
                frames.append(loaded_frame / 255 if normalize else loaded_frame)
            except Exception as err:
                print(f"*"*80)
                print(f"Failed for videoId = {row["videoId"]}, frameNr = {row["frameNr"]}")
                print(str(err))
                print(f"*"*80)
        frames = np.array(frames)
        if self.as_video:
            frames = np.expand_dims(frames, axis=0)
        return frames, np.array(y_values, dtype=np.float32)

    def on_epoch_end(self):
        self.Frames = self.Frames.sample(frac=1.)
