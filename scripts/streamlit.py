import streamlit as st
import numpy as np
import pandas as pd


pages = {
    "Your account": [
        st.Page("pages/browse.py", title="Browse"),
        st.Page("pages/label.py", title="Label"),
        st.Page("pages/record.py", title="Record"),
        st.Page("pages/stats.py", title="Stats"),
        st.Page("pages/settings.py", title="Settings"),
    ],
}

pg = st.navigation(pages)
pg.run()