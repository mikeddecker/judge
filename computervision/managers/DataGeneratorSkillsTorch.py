import keras
import numpy as np
import pandas as pd
import random
import math
import sys
import torch
from helpers import load_batch_X_torch, load_batch_y_torch, adaptSkillLabels
sys.path.append('..')
from .DataRepository import DataRepository
from .FrameLoader import FrameLoader
sys.path.append('../..')
from api.helpers import ConfigHelper
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# TODO : change to tf dataset, so prefetch is possible https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGeneratorSkills(torch.utils.data.Dataset):
    def __init__(self,
                 frameloader: FrameLoader,
                 train_test_val: str, # train, test, val
                 dim: tuple, # e.g. (128,128)
                 timesteps=16,
                 batch_size=1,
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

        self.balancedType = 'jump_return_push_frog_other' # TODO : make dynamic, provide in init
        self.Skills = adaptSkillLabels(df_skills=self.Skills, balancedType=self.balancedType)
        self.BalancedSet = pd.DataFrame(columns=self.Skills.columns)

        self.info_columns = [
            'type', 'rotations', 'turner1', 'turner2', 'skill', 'turntable',
            'bodyRotations', 'hands', 'feet', 'sloppy', 'hard2see', 'fault',
            'backwards'
        ]

        # Shuffle skills
        self.Skills = self.Skills.sample(frac=1.)
        self.__refillBalancedSet()
        print('DataGeneratorSkills init done')
        print(self.Skills["skill"].value_counts())
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.BalancedSet) // self.batch_size if self.train_test_val == 'train' else len(self.Skills) // self.batch_size

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        
        skillinfo_row = self.BalancedSet.iloc[batch_nr] if self.train_test_val == 'train' else self.Skills.iloc[batch_nr]
        videoId = skillinfo_row["videoId"]
        frameStart = skillinfo_row["frameStart"]
        frameEnd = skillinfo_row["frameEnd"]

        if batch_nr + 1 == self.__len__():
            self.on_epoch_end()

        X = load_batch_X_torch(
            frameloader=self.frameloader,
            videoId=videoId,
            dim=self.dim,
            frameStart=frameStart,
            frameEnd=frameEnd,
            augment=True if self.train_test_val == 'train' and normalize else False,
            timesteps=self.timesteps,
            normalized=normalize,
        )
        y = load_batch_y_torch(skillinfo_row=skillinfo_row, flip_turner=False)
        return X, y

    def on_epoch_end(self):
        self.Skills = self.Skills.sample(frac=1.)
        self.__refillBalancedSet()
        # print("@"*80)
        # print("@"*80)
        # print("@"*80)
    
    # def __get_multiplier(self, occurance_percentage: float, max_occurance_percentage: float, N = 10):
    #     """Calculate how many times more a skill needs to be added
    #     Returns a number in the interval [1, N["""
    #     assert isinstance(occurance_percentage, (int, float))
    #     assert isinstance(max_occurance_percentage, (int, float))
    #     assert occurance_percentage > 0 and occurance_percentage <= 1
    #     assert max_occurance_percentage > 0 and max_occurance_percentage <= 1
    #     assert max_occurance_percentage >= occurance_percentage, f"got {max_occurance_percentage} and {occurance_percentage}"

    #     multiplier = N * np.pow(1 - (np.sqrt(occurance_percentage / max_occurance_percentage)), 2)
    #     multiplier = multiplier if multiplier >= 1 else multiplier + 1
    #     multiplier_squared = multiplier * multiplier
    #     return multiplier_squared

    def __refillBalancedSet(self):        
        skillValueCounts = self.Skills["skill"].value_counts()
        lowestTrainAmount = min(
            skillValueCounts.loc[1], # Jumps
            skillValueCounts.loc[2], # Returns
            skillValueCounts.loc[3], # Pushups
            skillValueCounts.loc[4], # Frogs
            skillValueCounts.loc[5], # other
        )
        # print(skillValueCounts)

        self.BalancedSet = pd.concat([
            self.Skills[self.Skills['skill'] == 1].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 2].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 3].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 4].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 5].iloc[:lowestTrainAmount]
        ], ignore_index=True)
        self.BalancedSet = self.BalancedSet.sample(frac=1.)