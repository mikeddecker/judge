import os
import sys
from dotenv import load_dotenv
load_dotenv()

print("Loaded .env file")

STORAGE_DIR_VIDEOS = os.getenv("STORAGE_DIR_VIDEOS") 
STORAGE_DIR_BACKUPS = os.getenv("MYSQL_BACKUP")
STORAGE_DIR_GENERATED_DATA = os.getenv("STORAGE_DIR_GENERATED_DATA")

for dir_path in [STORAGE_DIR_VIDEOS, STORAGE_DIR_BACKUPS, STORAGE_DIR_GENERATED_DATA]:
    if dir_path is None:
        print(f"❌ Environment variable for storage directory not set properly.")
        sys.exit(1)
    
    os.system(f"rm -rf {dir_path}/*") if os.path.exists(dir_path) else os.makedirs(STORAGE_DIR_VIDEOS, exist_ok=True)

