import keras
import numpy as np
import pandas as pd
from DataRepository import DataRepository
from FrameLoader import FrameLoader
import sys
sys.path.append('..')
from api.helpers import ConfigHelper

# TODO : change to tf dataset, so prefetch is possible https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGeneratorSkills(keras.utils.Sequence):
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
        self.Skills = self.repo.get_skills(train_test_val)
        self.SkillCounts = self.repo.get_skill_category_counts()


        print('DataGeneratorSkills init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.Skills) // self.batch_size

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        skillinfo_row = self.Skills.iloc[batch_nr]
        videoId = skillinfo_row["videoId"]
        frameStart = skillinfo_row["frameStart"]
        frameEnd = skillinfo_row["frameEnd"]

        try:
            loaded_frames = self.frameloader.get_skill(videoId, self.dim, 
                                                    start=frameStart, 
                                                    end=frameEnd,
                                                    timesteps=self.timesteps, 
                                                    normalized=normalize)
            loaded_frames = np.expand_dims(loaded_frames, axis=0)
            y = {}
            for key, value in ConfigHelper.get_discipline_DoubleDutch_config().items():
                if key == "Tablename":
                    continue
                key_lower = key[0].lower() + key[1:]
                y[key] = skillinfo_row[key_lower]

                if value[0] == "Categorical":
                    y[key] -= 1
                elif value[0] == "Numerical":
                    y[key] /= value[2]

                y[key] = np.expand_dims(np.array(y[key]), axis=0)

        except Exception as err:
            print(f"*"*80)
            print(f"Failed for videoId = {videoId}, skillId = {skillinfo_row["id"]}")
            print(str(err))
            print(f"*"*80)

        return loaded_frames, y

    def on_epoch_end(self):
        self.Skills = self.Skills.sample(frac=1.)
