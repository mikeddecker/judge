# Run the api 🧠

## Run app

Everything is now dockerized! Just follow the [main README setup](../README.md) setup.

The API will be available at `http://localhost:${API_LOCAL_PORT}`

Database migrations and setup are handled automatically in the docker containers.

## Changing the databse (Flask - alembic)

(this has not been tried in the dockerized environment yet!)

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

## Test flow

Flow of the tests:

`__init__.py`
...

## FAQ

### Can FrameNr = 0?

According to my tests and frameInfo.py, yes it can.
Currently wondering wether it should be like that.

