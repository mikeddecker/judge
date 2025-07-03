import math
import numpy as np
import pandas as pd
import random
import sys
import torch
from .DataRepository import DataRepository
from .FrameLoader import FrameLoader
from helpers import calculate_splitpoint_values, load_segment_batch_X_torch, load_segment_batch_y_torch

sys.path.append('..')
from api.helpers import ConfigHelper

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO : change to tf dataset, so prefetch is possible https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGeneratorSegmentation(torch.utils.data.Dataset):
    def __init__(self,
                 frameloader: FrameLoader,
                 train_test_val: str, # train, test, val
                 dim: tuple, # e.g. (128,128)
                 timesteps=16,
                 batch_size=1,
                 normalized=True,
                 **kwargs):
        super().__init__(**kwargs)
        assert isinstance(dim, tuple)
        assert isinstance(dim[0], int)
        assert isinstance(dim[1], int)
        assert len(dim) == 2
        assert isinstance(timesteps, int)
        assert isinstance(train_test_val, str)
        assert train_test_val in ['train', 'test', 'val']
        self.dim = dim
        self.train_test_val = train_test_val
        self.augment = train_test_val == 'train'
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.frameloader = frameloader
        self.repo = DataRepository()
        self.Videos = self.repo.get_fully_segmented_videos(train_test_val)
        self.Skills = self.repo.get_skills_of_fully_segmented_videos(train_test_val)
        self._create_df_segment_samples()

        print('DataGeneratorSkills init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.df_batches)

    def __getitem__(self, batch_nr, normalize=False):
        "batch_nr starts from 0"
        
        segmentinfo_row = self.df_batches.iloc[batch_nr]

        videoId = segmentinfo_row["videoId"]
        frameStart = segmentinfo_row["frameStart"]
        frameEnd = frameStart + self.timesteps

        try:
            X = load_segment_batch_X_torch(
                frameloader=self.frameloader,
                videoId=videoId,
                dim=self.dim,
                timesteps=self.timesteps,
                frameStart=frameStart,
                frameEnd=frameEnd,
                normalized=normalize
            )
            y = load_segment_batch_y_torch(
                frameStart=frameStart,
                frameEnd=frameEnd,
                df_splitpoint_values=self.df_frames[self.df_frames['videoId'] == videoId]
            )

        except Exception as err:
            print(f"*"*80)
            print(f"Failed for videoId = {videoId}, skillId = {segmentinfo_row}, {batch_nr}/{self.__len__()}")
            print(str(err))
            print(f"*"*80)
            raise err

        return X, y

    def on_epoch_end(self):
        self._create_df_segment_samples() # Recreate, as a random offset is taken

    def _create_df_segment_samples(self):
        dfs = []
        dfs_train_batches = []

        offset = random.randint(0, self.timesteps)

        for _, row in self.Videos.iterrows():
            videoId = row['id']
            frameLength = row['frameLength']

            df = calculate_splitpoint_values(
                videoId=videoId,
                frameLength=frameLength,
                df_Skills=self.Skills[self.Skills['videoId'] == videoId],
                fps = row["fps"])
            
            dfs.append(df)
            
            # Batches
            n_batches_video = (len(df) - offset) // self.timesteps
            trainvalues = {
                'videoId' : [videoId for _ in range(n_batches_video)],
                'frameStart' : [offset + i * self.timesteps for i in range(n_batches_video)]
            }
            dfs_train_batches.append(pd.DataFrame(trainvalues))


        self.df_frames = pd.concat(dfs)
        self.df_batches = pd.concat(dfs_train_batches).sample(frac=1.)