# Run the api

## Flask

```bash
flask db migrate -m "Initial migration"  # Generates the migration script
flask db upgrade         # Applies the migration to the database
```

## Run app

```bash
python app.py
```
