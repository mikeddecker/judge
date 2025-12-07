# Run the api

## Run app

Fill in the .env

```env
MYSQL_DATABASE = judge_db
MYSQL_DATABASE_TEST = judge_test
MYSQL_USERNAME = root
MYSQL_ROOT_PASSWORD = root
MYSQL_LOCAL_PORT = 3377
MYSQL_DOCKER_PORT = 3306
MYSQL_HOST = mysqldb

# The directory where MYSQL backups go
MYSQL_BACKUP = /media/miked/Elements/Judge/results/backups

# The directory where all videos are stored. They can be subcategorized in folders.
STORAGE_DIR_VIDEOS = /media/miked/Elements/Judge/videos

# The directory where all the app's generated data will come.
STORAGE_DIR_GENERATED_DATA = /media/miked/Elements/Judge/results

# Directory for test purposes
TESTDIR = /tmp/judge

# Connection strings for connecting with the database
# They are based on the input data above, keep off.
DATABASE_URL = mysql+pymysql://${MYSQL_USERNAME}:${MYSQL_ROOT_PASSWORD}@${MYSQL_HOST}:${MYSQL_DOCKER_PORT}/${MYSQL_DATABASE}
DATABASE_URL_TEST =  mysql+pymysql://${MYSQL_USERNAME}:${MYSQL_ROOT_PASSWORD}@${MYSQL_HOST}:${MYSQL_DOCKER_PORT}/${MYSQL_DATABASE_TEST}

# Ports used by the API service
# local port on your machine, docker port inside the container
API_LOCAL_PORT=5555
API_DOCKER_PORT=5555
WEB_LOCAL_PORT=5173
WEB_DOCKER_PORT=5173

# Video data you want to support.
SUPPORTED_VIDEO_FORMATS = ['.mov', '.mp4', '.MP4']
SUPPORTED_IMAGE_FORMATS = ['.jpeg', '.png']
```

Prerequisites
- Install Docker (& Docker compose)
- Check [post installation](https://docs.docker.com/engine/install/linux-postinstall/) of docker

1. Install python requirements `pip install -r requirements.txt`. MYSQLCLIENT_CFLAGS and MYSQLCLIENT_LDFLAGS error? see fix below.
1. Start the docker container: `docker compose up -d`
2. Create an empty database with .env same name as `MYSQL_DATABASE`
3. Run the database migrations: `flask db upgrade`
4. Run the app: `python app.py`
5. Check if you can run `http://localhost:${API_LOCAL_PORT}/folders` in your web browser (if not, check port forwarding e.g. `app.py` - `app.run(port=API_DOCKER_PORT, debug=True)`)

## Mysql error?

When starting up the docker, it won't install mysql?
Download the needed mysql packages:

```bash
sudo apt-get install default-libmysqlclient-dev build-essential pkg-config
```

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

