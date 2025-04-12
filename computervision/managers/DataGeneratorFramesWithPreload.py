import keras
import numpy as np
import pandas as pd
from DataRepository import DataRepository

# Didn't improve loading time

class DataGeneratorFramesWithPreload(keras.utils.Sequence):
    def __init__(self,
                 frameloader,
                 train_test_val: str, # train, test, val
                 dim: tuple, # e.g. (128,128)
                 batch_size=32, # Default batch size
                 normalized=True,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dim, tuple)
        assert len(dim) == 2
        assert isinstance(train_test_val, str)
        assert train_test_val in ['train', 'test', 'val']
        self.dim = dim
        self.train_test_val = train_test_val
        self.batch_size = batch_size
        self.frameloader = frameloader
        self.repo = DataRepository()
        self.Frames = self.repo.get_framelabels(train_test_val)
        self.Frames['videoId'] = self.Frames['videoId'].astype(int)
        self.Frames['frameNr'] = self.Frames['frameNr'].astype(int)
        self.preload_batches = 16

        print('DataGeneratorSkillBorders init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.Frames) // self.batch_size + 1
    
    def __preload_items(self, item, batches):
        'For the chance that a video has in the near neighbourhood other frames to be loaded, only once open this video'
        min_idx = item * self.batch_size * batches
        max_idx = item * self.batch_size * batches + self.batch_size * batches

        # On sampling:'videoId' : { 'frameNr' : (x,y,w,h) }
        # On loading: 'videoId' : { 'frameNr' : (loaded_frame, y) }
        self.preloaded = {}

        # Fill preloaded keys with frames to load
        for i in range(min_idx, max_idx):
            row = self.Frames.iloc[i]
            videoId = int(row["videoId"])
            frameNr = int(row["frameNr"])
            x, y, w, h = self.Frames.iloc[i][["x", "y", "width", "height"]]
            
            if videoId not in self.preloaded.keys():
                self.preloaded[videoId] = { frameNr: (x,y,w,h) }
            else:
                self.preloaded[videoId][frameNr] = (x,y,w,h)
        
        for videoId in self.preloaded.keys():
            # Loading images
            try:
                loaded_frames = self.frameloader.get_frames(videoId, self.preloaded[videoId], self.dim[0])
                self.preloaded[videoId] = loaded_frames
            except Exception as err:
                print(f"*"*80)
                print(f"Failed loading for videoId = {videoId}, frameNr = {self.preloaded[videoId].keys()}")
                print(str(err))
                print(f"*"*80)
        
    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        if batch_nr % self.preload_batches == 0:
            print("Batch_nr", batch_nr, "preloading")
            self.__preload_items(batch_nr % self.preload_batches, self.preload_batches)
        
        frames = []
        i_start = batch_nr * self.batch_size
        i_end = min(len(self.Frames), (batch_nr + 1) * self.batch_size)
        y_values = []
        for i in range(i_start, i_end):
            row = self.Frames.iloc[i]
            videoId = row["videoId"].astype(int)
            frameNr = row["frameNr"].astype(int)
            f, ys = self.preloaded[videoId][frameNr]
            frames.append(f / 255 if normalize else f)
            y_values.append(ys)
        return np.array(frames), np.array(y_values, dtype=np.float32)


    def on_epoch_end(self):
        self.Frames = self.Frames.sample(frac=1.)
