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
        self.BalancedSkills = pd.DataFrame(columns=self.Skills.columns)
        self.info_columns = [
            'type', 'rotations', 'turner1', 'turner2', 'skill', 'turntable',
            'bodyRotations', 'hands', 'feet', 'sloppy', 'hard2see', 'fault',
            'backwards'
        ]

        print('DataGeneratorSkills init done')
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.BalancedSkills7) // self.batch_size

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        skillinfo_row = self.BalancedSkills7.iloc[batch_nr]
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
        self.BalancedSkills3 = self.__balance_skills(N=3)
        self.BalancedSkills5 = self.__balance_skills(N=5)
        self.BalancedSkills7 = self.__balance_skills(N=7)
        self.BalancedSkills10 = self.__balance_skills(N=10)
        self.BalancedSkills12 = self.__balance_skills(N=12)
        self.BalancedSkills7 = self.BalancedSkills7.sample(frac=1.)
    
    def __get_multiplier(self, occurance_percentage: float, max_occurance_percentage: float, N = 10):
        """Calculate how many times more a skill needs to be added
        Returns a number in the interval [1, N["""
        assert isinstance(occurance_percentage, (int, float))
        assert isinstance(max_occurance_percentage, (int, float))
        assert occurance_percentage > 0 and occurance_percentage <= 1
        assert max_occurance_percentage > 0 and max_occurance_percentage <= 1
        assert max_occurance_percentage >= occurance_percentage, f"got {max_occurance_percentage} and {occurance_percentage}"

        multiplier = N * np.pow(1 - (np.sqrt(occurance_percentage / max_occurance_percentage)), 2)
        multiplier = multiplier if multiplier >= 1 else multiplier + 1
        multiplier_squared = multiplier * multiplier
        return multiplier_squared


    def __balance_skills(self, N=7):        
        balanced_skills = pd.DataFrame(columns=self.Skills.columns)
        for col in self.info_columns:
            series_normalized_occurances = self.Skills[col].value_counts(normalize=True)
            max_occurance_percentage = series_normalized_occurances.iloc[0]

            for index, value in series_normalized_occurances.items():
                if value == max_occurance_percentage:
                    continue
                rounded_multiplier = int(np.round(self.__get_multiplier(value, max_occurance_percentage, N=N)))
                extra_skills = [self.Skills[self.Skills[col] == index] for _ in range(rounded_multiplier)]
                extra_skills.append(balanced_skills)
                balanced_skills = pd.concat(extra_skills, ignore_index=True)
        return balanced_skills
