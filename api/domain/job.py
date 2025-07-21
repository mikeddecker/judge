from datetime import datetime
from helpers.ValueHelper import ValueHelper
from config import JOB_STEPS, JOB_TYPES

class Job:
    def __init__(
            self,
            type: str,
            step: str,
            status: str = 'Created',
            id: int = None,
            job_arguments: dict = {},
            request_time: datetime = None,
            status_details: str = None,
        ):

        assert type in JOB_TYPES, f'JobType ({type}) must be in {JOB_TYPES}'
        assert step in JOB_STEPS, f'Job step ({step}) must be in {JOB_STEPS}'
        if type == 'PREDICT':
            assert 'videoId' in job_arguments.keys(), f"VideoId must be specified for predict jobs"
            assert job_arguments['videoId'] is not None, f"VideoId may not be None for predict jobs"
            ValueHelper.check_raise_id(job_arguments['videoId'])

        if type == 'TRAIN' or type == 'PREDICT':
            assert 'model' in job_arguments.keys(), f"Model must be specified for predict or train jobs"
            assert job_arguments['model'] is not None, f"Model may not be None for predict or train jobs"
            ValueHelper.check_raise_string(job_arguments['model'])

        self.id = id
        self.type = type
        self.step = step
        self.job_arguments = job_arguments
        self.request_time = request_time if request_time else datetime.now()
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