from datetime import datetime

class Job:
    def __init__(
            self,
            type: str,
            step: str,
            status: str = 'Waiting',
            id: int = None,
            job_arguments: dict = {},
            request_time: datetime = None,
            status_details: str = None,
        ):
        self.id = id
        self.type = type
        self.step = step
        self.job_arguments = job_arguments
        self.request_time = request_time
        self.status = status
        self.status_details = status_details
    
    def to_dict(self):
        return {
            'id' : self.id,
            'type' : self.type,
            'step' : self.step,
            'job_arguments' : self.job_arguments,
            'request_time' : self.request_time,
            'status' : self.status,
            'status_details' : self.status_details,
        }