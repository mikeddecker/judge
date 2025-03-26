import keras
import numpy as np
import pandas as pd
import random
import math
from DataRepository import DataRepository
from FrameLoader import FrameLoader
import sys
sys.path.append('..')
from api.helpers import ConfigHelper

# TODO : change to tf dataset, so prefetch is possible https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGeneratorSegmentation(keras.utils.Sequence):
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
        return len(self.df_samples)

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        
        segmentinfo_row = self.df_samples.iloc[batch_nr]

        videoId = segmentinfo_row["videoId"]
        frameStart = segmentinfo_row["frameStart"]
        frameEnd = frameStart + self.timesteps

        try:
            loaded_frames = self.frameloader.get_segment(videoId, self.dim, 
                                                    start=frameStart, 
                                                    end=frameEnd,
                                                    normalized=normalize,
                                                    )
            loaded_frames = np.expand_dims(loaded_frames, axis=0)
            y = self.df_frames[(self.df_frames['videoId'] == videoId) & (self.df_frames['frameNr'] >= frameStart) & (self.df_frames['frameNr'] < frameEnd)]['splitpoint'].to_numpy()
            y = np.expand_dims(y, axis=0)

        except Exception as err:
            print(f"*"*80)
            print(f"Failed for videoId = {videoId}, skillId = {segmentinfo_row["id"]}")
            print(str(err))
            print(f"*"*80)
            raise err

        return loaded_frames, y

    def on_epoch_end(self):
        self._create_df_segment_samples() # Recreate, as a random offset is taken

    def _create_df_segment_samples(self):
        dfs = []
        dfs_train_samples = []

        offset = random.randint(0, self.timesteps)
        Nsec_frames_around = 0.14

        for _, row in self.Videos.iterrows():
            videoId = row['id']
            frameLength = row['frameLength']

            splitpoint_values = {
                'videoId' : [videoId for _ in range(frameLength)],
                'frameNr' : range(frameLength),
                'splitpoint' : [0 for _ in range(frameLength)],
            }



            frames_around_splitpoint = round(Nsec_frames_around * row['fps'])
            for _, skillrow in self.Skills[self.Skills["videoId"]==videoId].iterrows():
                frameStart = skillrow["frameStart"]
                frameEnd = skillrow["frameEnd"]

                currentFrameStart = frameStart - frames_around_splitpoint
                currentFrameEnd = frameEnd - frames_around_splitpoint
                while currentFrameStart < frameStart + frames_around_splitpoint:
                    framesApart = abs(currentFrameStart - frameStart)
                    splitvalue = 1 - (framesApart / frames_around_splitpoint) ** 2
                    splitvalue *= splitvalue

                    currentFrameStart += 1
                    currentFrameEnd += 1

                    splitpoint_values['splitpoint'][currentFrameStart] = splitvalue
                    if currentFrameEnd < frameLength:
                        splitpoint_values['splitpoint'][currentFrameEnd] = splitvalue


            df = pd.DataFrame(splitpoint_values)
            samples = (len(df) - offset) // self.timesteps

            dfs.append(df.iloc[offset:offset+samples*self.timesteps])

            trainvalues = {
                'videoId' : [videoId for _ in range(samples)],
                'frameStart' : [offset + i * self.timesteps for i in range(samples)]
            }

            dfsample = pd.DataFrame(trainvalues)
            dfs_train_samples.append(dfsample)


        self.df_frames = pd.concat(dfs)
        self.df_samples = pd.concat(dfs_train_samples).sample(frac=1.)