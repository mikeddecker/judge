import keras
import numpy as np
import pandas as pd
import random
import math
import sys
import torch
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

        self.__enrichSkills()
        self.balancedType = 'jump_return_push_frog_other' # TODO : make dynamic
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
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.BalancedSet) // self.batch_size # if self.train_test_val == 'train' else len(self.Skills) // self.batch_size

    def __getitem__(self, batch_nr, normalize=True):
        "batch_nr starts from 0"
        
        skillinfo_row = self.BalancedSet.iloc[batch_nr] # if self.train_test_val == 'train' else self.Skills.iloc[batch_nr]
        videoId = skillinfo_row["videoId"]
        frameStart = skillinfo_row["frameStart"]
        frameEnd = skillinfo_row["frameEnd"]

        try:
            loaded_frames, flip_turner = self.frameloader.get_skill_torch(videoId, self.dim, 
                                                    start=frameStart, 
                                                    end=frameEnd,
                                                    timesteps=self.timesteps, 
                                                    normalized=normalize,
                                                    augment=True if self.train_test_val == 'train' and normalize else False,
                                                    flip_image=False,
                                                    # flip_image=(normalize and self.train_test_val == 'train' and random.random() < 0.5)
                                                    )
            loaded_frames = torch.from_numpy(loaded_frames).float().to(device)  # [timesteps, C, H, W]
            
            # Prepare targets - no batch dimension needed
            y = {}
            for key, value in ConfigHelper.get_discipline_DoubleDutch_config().items():
                if key == "Tablename":
                    continue
                    
                key_lower = key[0].lower() + key[1:]
                if flip_turner and key in ["Turner1", "Turner2"]:
                    key = "Turner2" if key == "Turner1" else "Turner1"

                target_value = skillinfo_row[key_lower]
                
                if value[0] == "Categorical":
                    # Convert to 0-based index and long tensor
                    y[key] = torch.tensor(int(target_value) - 1, dtype=torch.long).to(device)
                elif value[0] == "Numerical":
                    # Normalize and convert to float tensor
                    normalized_value = target_value / value[2]
                    y[key] = torch.tensor(normalized_value, dtype=torch.float).to(device)
                else:  # Boolean flags
                    y[key] = torch.tensor(bool(target_value), dtype=torch.float).to(device)
                    
        except Exception as err:
            print(f"*"*80)
            print(f"Failed for videoId = {videoId}, skillId = {skillinfo_row["id"]}")
            print(str(err))
            print(f"*"*80)
            raise err

        if batch_nr + 1 == self.__len__():
            self.on_epoch_end() # Just in case 4 now

        return loaded_frames, y

    def on_epoch_end(self):
        self.Skills = self.Skills.sample(frac=1.)
        self.__refillBalancedSet()
        print("@"*80)
        print("@"*80)
        print("@"*80)
    
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

    def __enrichSkills(self):
        self.Skills['skill'] = np.where(
            self.Skills['skill'] <= 5,
            self.Skills['skill'],
            5
        )
    def __refillBalancedSet(self):        
        skillValueCounts = self.Skills["skill"].value_counts()
        lowestTrainAmount = min(
            skillValueCounts.loc[1], # Jumps
            skillValueCounts.loc[2], # Returns
            skillValueCounts.loc[3], # Pushups
            skillValueCounts.loc[4], # Frogs
            skillValueCounts.loc[5], # other
        )
        print(skillValueCounts)

        self.BalancedSet = pd.concat([
            self.Skills[self.Skills['skill'] == 1].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 2].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 3].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 4].iloc[:lowestTrainAmount],
            self.Skills[self.Skills['skill'] == 5].iloc[:lowestTrainAmount]
        ], ignore_index=True)