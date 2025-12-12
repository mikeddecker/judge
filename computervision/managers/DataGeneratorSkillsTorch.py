import keras
import numpy as np
import pandas as pd
import random
import math
import sys
import torch

from models.OutputHeadRecognition import OutputHeadRecognition
from helpers import load_skill_batch_X_torch, load_skill_batch_y_torch, adaptSkillLabels

from managers.DataRepository import DataRepository
from managers.FrameLoader import FrameLoader

# TODO : change to tf dataset, so prefetch is possible https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
class DataGeneratorSkills(torch.utils.data.Dataset):
    def __init__(self,
                 frameloader: FrameLoader,
                 head: OutputHeadRecognition,
                 train_test_val: str, # train, test, val
                 dim: tuple, # e.g. (128,128)
                 timesteps=16,
                 batch_size=1,
                 testrun: bool = False,
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
        self.isTestrun = testrun
        self.frameloader = frameloader
        self.repo = DataRepository()
        self.Skills = self.repo.get_skills(train_test_val)
        self.head = head

        self.balancedType = 'jump_return_push_frog_other' # TODO : make dynamic, provide in init
        self.balancedType = 'limit_5procent'
        self.BalancedSet = pd.DataFrame(columns=self.Skills.columns)

        self.info_columns = [
            'type', 'rotations', 'turner1', 'turner2', 'skill', 'turntable',
            'bodyRotations', 'hands', 'feet', 'sloppy', 'hard2see', 'fault',
            'backwards'
        ]

        # Shuffle skills
        self.Skills = self.Skills.sample(frac=1.)
        self.__refillBalancedSet()
        print(f'DataGeneratorSkills {train_test_val} init done')
        self.on_epoch_end()

    def __len__(self, balanced=True):
        'Denotes the number of batches per epoch'
        if not balanced:
            return len(self.Skills)
        if self.isTestrun:
            return min(280, len(self.BalancedSet))
        return len(self.BalancedSet) // self.batch_size if self.train_test_val == 'train' else len(self.Skills) // self.batch_size

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"

        skillinfo_row = self.BalancedSet.iloc[batch_nr] if self.train_test_val == 'train' else self.Skills.iloc[batch_nr]
        videoId = skillinfo_row["videoId"]
        frameStart = skillinfo_row["frameStart"]
        frameEnd = skillinfo_row["frameEnd"]

        X, flip_turner = load_skill_batch_X_torch(
            frameloader=self.frameloader,
            videoId=videoId,
            dim=self.dim,
            frameStart=frameStart,
            frameEnd=frameEnd,
            augment=True if self.train_test_val == 'train' and normalize else False,
            timesteps=self.timesteps,
            normalized=normalize,
        )
        try:
            target, mask = self.head.label_to_tensor(skillinfo_row['skillinfo'])
        except:
            print(f"Error for skill: {skillinfo_row['id']}")
            raise

        return X, target, mask

    def on_epoch_end(self):
        self.Skills = self.Skills.sample(frac=1.)
        self.__refillBalancedSet()

    def __refillBalancedSet(self):
        self.BalancedSet = self.limit_skill_occurrences()

    def limit_skill_occurrences(self):
        if self.train_test_val != 'train':
            return self.Skills
        
        counts = self.Skills['skillinfo_string'].value_counts()
        max_allowed = int(round(counts.max() ** 0.5))

        limited_df = pd.concat([
            group.sample(n=min(len(group), max_allowed)) 
            for _, group in self.Skills.groupby('skillinfo_string')
        ])
        return limited_df.reset_index(drop=True)

