#!/usr/bin/env python
# coding: utf-8
import json
import os
import pandas as pd
import sqlalchemy as sqlal

from collections import defaultdict
from constants import ENVS

from managers.RepoGeneral import DataRepository


class RepoStats(DataRepository):
    VideoNames = {} # pandas dataframe

    def __init__(self, kwargs):
        super.__init__(**kwargs)
    
REPO_STATS = RepoStats()