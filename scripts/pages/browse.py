from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import streamlit as st

load_dotenv()

st.set_page_config(
    layout="wide"
)

# Working dir - SSD drive
main_dir = os.getenv("STORAGE_DIR")
current_dir = main_dir

folder_content = os.listdir(current_dir)
make_full_path = lambda f : os.path.join(current_dir, f)

folders = list(filter(lambda f : os.path.isdir(make_full_path(f)) , folder_content))
files = list(filter(lambda f : os.path.isfile(make_full_path(f)), folder_content))

####################
#  Display
####################


c = st.container(border=True)
c.write("Video1")
c.write("Video2")

st.write("Folders here")

folders

vids_per_row = st.slider("Vids per row", min_value=1, max_value=5, value=3)

groups = []
for i in range(0, len(files), vids_per_row):
    groups.append(files[i:i + vids_per_row])

for row in groups:
    cols = st.columns(vids_per_row, vertical_alignment="bottom")
    for i, file in enumerate(row):
        cols[i].image(os.path.join(current_dir, file))
        cols[i].text(file)
