import streamlit as st
import pandas as pd
import numpy as np

map_data = pd.DataFrame(
    np.random.randn(1000, 2) / [10, 2.4] + [51.0, 4.20],
    columns=['lat', 'lon'])

map_data