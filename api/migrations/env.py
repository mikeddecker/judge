import os
import logging
from logging.config import fileConfig

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')

def _get_flask_migrate():
    """Return Flask-Migrate instance if a Flask app context is active, else None."""
    try:
        from flask import current_app
        return current_app.extensions['migrate']
    except RuntimeError:
        return None

def _get_standalone_url():
    """Build a SQLAlchemy URL from environment variables (used outside Flask context)."""
    host = os.environ.get('MYSQL_HOST', 'localhost')
    _docker_port = os.environ.get('MYSQL_DOCKER_PORT')
    port = _docker_port if _docker_port is not None else os.environ.get('MYSQL_PORT', '3306')
    user = os.environ.get('MYSQL_USERNAME', 'root')
    password = os.environ.get('MYSQL_ROOT_PASSWORD', '')
    database = os.environ.get('MYSQL_DATABASE', '')
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"

def get_engine_url():
    migrate = _get_flask_migrate()
    if migrate is not None:
        try:
            return migrate.db.engine.url.render_as_string(hide_password=False).replace('%', '%%')
        except AttributeError:
            return str(migrate.db.engine.url).replace('%', '%%')
    return _get_standalone_url()

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
config.set_main_option('sqlalchemy.url', get_engine_url())

def get_metadata():
    migrate = _get_flask_migrate()
    if migrate is None:
        return None
    target_db = migrate.db
    if hasattr(target_db, 'metadatas'):
        return target_db.metadatas[None]
    return target_db.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.

def run_migrations_offline():
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=get_metadata(), literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """

    # this callback is used to prevent an auto-migration from being generated
    # when there are no changes to the schema
    # reference: http://alembic.zzzcomputing.com/en/latest/cookbook.html
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    migrate = _get_flask_migrate()
    if migrate is not None:
        conf_args = migrate.configure_args
        if conf_args.get("process_revision_directives") is None:
            conf_args["process_revision_directives"] = process_revision_directives
        connectable = migrate.db.engine
    else:
        from sqlalchemy import engine_from_config, pool
        conf_args = {"process_revision_directives": process_revision_directives}
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix='sqlalchemy.',
            poolclass=pool.NullPool,
        )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            compare_indexes=False, # TODO : rename foreign key propertyId
            **conf_args
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

