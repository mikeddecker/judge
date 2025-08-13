# Run the api

## Run app

Fill in the .env

```env
MYSQLDB_DATABASE = judge_db
MYSQLDB_DATABASE_TEST = judge_test
MYSQLDB_ROOT_PASSWORD = root
MYSQLDB_LOCAL_PORT = 3377
MYSQLDB_DOCKER_PORT = 3306

# The directory where all videos are stored. They can be subcategorized in folders.
STORAGE_DIR_VIDEOS = /media/miked/Elements/Judge/videos

# The directory where all the app's generated data will come.
STORAGE_DIR_GENERATED_DATA = /media/miked/Elements/Judge/results

# Directory for test purposes
TESTDIR = /tmp/judge

# Connection strings for connecting with the database
# They are based on the input data above, keep off.
DATABASE_URL = mysql+pymysql://root:${MYSQLDB_ROOT_PASSWORD}@127.0.0.1:${MYSQLDB_LOCAL_PORT}/${MYSQLDB_DATABASE}
DATABASE_URL_TEST =  mysql+pymysql://root:${MYSQLDB_ROOT_PASSWORD}@127.0.0.1:${MYSQLDB_LOCAL_PORT}/${MYSQLDB_DATABASE_TEST}

# Video data you want to support.
SUPPORTED_VIDEO_FORMATS = ['.mov', '.mp4', '.MP4']
SUPPORTED_IMAGE_FORMATS = ['.jpeg', '.png']
```

1. Start the docker container: `docker compose up -d`
2. Create an empty database with .env same name as `MYSQLDB_DATABASE`
3. Run the database migrations: `flask db upgrade`
4. Run the app: `python app.py`

## Changing the databse (Flask - alembic)

```bash
flask db migrate -m "Initial migration"  # Generates the migration script for changes to the database
flask db upgrade         # Applies the migration to the database
```

## Creating a backup

```bash
mysqldump -u root -p -h 127.0.0.1 -P 3377 judge > /media/miked/Elements/Judge/FINISHED-DB-READY/$(date +%Y%m%d)_judge_dump.sql
```

## Restoring a backup

```bash
mysql -u root -p -h 127.0.0.1 -P 3377 judge < /media/miked/Elements/Judge/FINISHED-DB-READY/20250216_judge_dump.sql
```

