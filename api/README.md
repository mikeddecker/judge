# Run the api

`fastapi dev api.py --port 8123`

## TODO

Check for enum (folder paths)

## Prisma

1. Set the DATABASE_URL in the .env file to point to your existing database. If your database has no tables yet, [read](https://pris.ly/d/getting-started)
2. Run `prisma migrate dev` to set up the db (or ask a current version)
3. Run `prisma generate` to generate the Prisma Client. You can then start querying your database.
4. Tip: Explore how you can extend the ORM with scalable connection pooling, global caching, and real-time database events. [Read](https://pris.ly/beyond-the-orm)

More information in our documentation:
[read](https://pris.ly/d/getting-started)

### Alter schema

1. Update `schema.prisma`
2. `prisma migrate dev`
3. `prisma generate`
