# Imports

import time
from managers.DataRepository import DataRepository

# Managers

REPO = DataRepository()


# JobReader

no_shutdown_job = True

while no_shutdown_job:
    job = REPO.get_next_job()

    print("job is,", job)

    if job is None:
        time.sleep(3)
        continue

    print("exec")

    # Update, remove job
    




