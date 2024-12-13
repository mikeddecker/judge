import sys
import os
from dotenv import load_dotenv
load_dotenv()

STORAGE_DIR_TEST = os.getenv("STORAGE_DIR_TEST") 

if os.path.exists(STORAGE_DIR_TEST):
    os.system(f"rm -rf {STORAGE_DIR_TEST}/*")
else:
    os.mkdir(STORAGE_DIR_TEST)