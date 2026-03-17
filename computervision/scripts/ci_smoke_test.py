import json
import os
import time
from urllib.error import URLError
from urllib.request import urlopen

import sqlalchemy as sqlal

from constants import ENVS, RECIPES
from managers.RepoGeneral import RepoGeneral

def wait_for_api(api_url: str, attempts: int = 30, delay_seconds: int = 2) -> None:
    for _ in range(attempts):
        try:
            with urlopen(api_url, timeout=5) as response:
                if response.status == 200:
                    return
        except URLError:
            pass
        time.sleep(delay_seconds)
    raise RuntimeError(f"API did not become ready: {api_url}")

def wait_for_database(engine: sqlal.Engine, attempts: int = 30, delay_seconds: int = 2) -> None:
    for _ in range(attempts):
        try:
            with engine.connect() as connection:
                connection.execute(sqlal.text("SELECT 1"))
                return
        except Exception:
            time.sleep(delay_seconds)
    raise RuntimeError("Database did not become ready")

def main() -> None:
    api_host = os.getenv("API_HOST", "api")
    api_port = os.getenv("API_DOCKER_PORT", "5555")
    api_health_url = f"http://{api_host}:{api_port}/health"
    wait_for_api(api_health_url)

    database_connection = (
        f"mysql+pymysql://{ENVS.DATABASE.MYSQL_USERNAME}:{ENVS.DATABASE.MYSQL_ROOT_PASSWORD}"
        f"@{ENVS.DATABASE.MYSQL_HOST}:{ENVS.DATABASE.MYSQL_DOCKER_PORT}/{ENVS.DATABASE.MYSQL_DATABASE}"
    )
    engine = sqlal.create_engine(database_connection, pool_recycle=30)
    wait_for_database(engine)

    if not RECIPES:
        raise RuntimeError("RECIPES failed to load")

    repo = RepoGeneral()
    marker = "ci-cv-smoke"
    payload = json.dumps({"videoId": "00000000-0000-0000-0000-000000000000", "model": "dummy"})

    with repo._get_connection() as connection:
        insert_job = sqlal.text(
            """
            INSERT INTO Jobs (id, type, step, job_arguments, status, status_details, job_category, createdAt, updatedAt)
            VALUES (UUID_TO_BIN(UUID()), :type, :step, CAST(:job_arguments AS JSON), :status, :status_details, :job_category, NOW(), NOW())
            """
        )
        connection.execute(
            insert_job,
            {
                "type": "PREDICT",
                "step": "LOCALIZE",
                "job_arguments": payload,
                "status": "Created",
                "status_details": marker,
                "job_category": "AI",
            },
        )
        connection.commit()

        inserted_row = connection.execute(
            sqlal.text("SELECT id FROM Jobs WHERE status_details = :marker LIMIT 1"),
            {"marker": marker},
        ).fetchone()

    if inserted_row is None:
        raise RuntimeError("Failed to insert dummy job")

    _ = repo.get_next_job()
    repo.delete_job(inserted_row[0])

    with repo._get_connection() as connection:
        remaining = connection.execute(
            sqlal.text("SELECT COUNT(*) FROM Jobs WHERE status_details = :marker"),
            {"marker": marker},
        ).scalar()

    if int(remaining or 0) != 0:
        raise RuntimeError("Dummy job was not cleaned up")

    print("Computervision CI smoke test passed (API + DB + job queue path)")

if __name__ == "__main__":
    main()

