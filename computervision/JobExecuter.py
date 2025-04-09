# Imports

import time
from DataRepository import DataRepository

# Managers

REPO = DataRepository()


# JobReader

no_shutdown_job = True

while no_shutdown_job:
    job = REPO.get_next_job()

    time.sleep(seconds=3)

    




