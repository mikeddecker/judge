import sys
import os
from dotenv import load_dotenv
load_dotenv()

TESTDIR = os.getenv("TESTDIR") 

if os.path.exists(TESTDIR):
    os.system(f"rm -rf {TESTDIR}/*")
else:
    os.mkdir(TESTDIR)