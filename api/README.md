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

## Create a discipline: specifiy skillinformation

Create a dictionary like below, followed by skillinfo (will be the numerical representations and unique constraints)
Create the actual tables: `Skillinfo_DoubleDutch_Type`, `Skillinfo_DoubleDutch_Skill`, `Skillinfo_DoubleDutch_Turner1`... in `models.py`

```python

config = {
    "Tablename" : "DoubleDutch",
    "Type" : ("Categorical"), # Will be textual representions
    "Skill" : ("Categorical"),
    "Rotations" : ("Numerical", minimum, maximum_included, step) # Must be integer, floats not yet included (in case 0.25 -> multiply by 4)
    "Hands" : ("Numerical", 1, 2, 1),
    "Turner1", ("Categorical"),
}

skillinfo = {
    "Type" : 1,
    "Skill" : 7,
    "Rotations" : 3,
    "Hands" : 2,
    "Turner1": 2,
    "Turner2": 2, # Will be ignored if not specified in config
}
```

So adding skill like boolean or numeric:
Add to ConfigHelper.py, mapToDomain, model.py, flask db migrate, and make sure frontend can accept the extra info.