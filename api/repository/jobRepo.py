from domain.job import Job
import json
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, func
from repository.models import Jobs as JobDB
from repository.MapToDomain import MapToDomain
from typing import List

class JobRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db
    
    def add(self, job: Job) -> None:
        new_job = JobDB(
            type = job.type,
            step = job.step,
            job_arguments = job.job_arguments,
            request_time = job.request_time,
            status = job.status,
            status_details = job.status_details,
        )
        self.db.session.add(new_job)
        self.db.session.commit()
        return MapToDomain.map_job(new_job)
    
    def exists(self, jobId: int) -> bool:
        return self.db.session.query(JobDB).filter_by(id=jobId).scalar() is not None
    
    def exists_by_job_content(self, job: Job) -> bool:
        return self.db.session.query(JobDB).filter(
            and_(
                JobDB.type == job.type, 
                JobDB.step == job.step,
                func.JSON_CONTAINS(JobDB.job_arguments, json.dumps(job.job_arguments))
            )
        ).count() > 0
    
    def get_all(self) -> List[Job]:
        """
        Returns all jobs
        """
        return [MapToDomain.map_job(j) for j in self.db.session.query(JobDB).all()]

    def delete(self, id: str) -> None:
        """
        Hard deletes the job from the database.
        """
        if not self.exists(id):
            raise LookupError(f"Folder {id} doesn't exist")
        jobdb = self.db.session.get(JobDB, ident=id)
        self.db.session.delete(jobdb)
        self.db.session.commit()


    def count(self) -> int:
        return self.db.session.query(JobDB).count()
