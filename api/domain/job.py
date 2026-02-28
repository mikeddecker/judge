from config import JOB_STEPS, JOB_TYPES
from datetime import datetime
from helpers.ValueHelper import ValueHelper
from uuid import UUID

class Job:
    def __init__(
            self,
            type: str,
            step: str,
            status: str = 'Created',
            id: UUID = None,
            job_arguments: dict = {},
            createdAt: datetime = None,
            status_details: str = None,
        ) -> None:
        f"""
        Docstring for __init__

        :param self: The Job instance
        :param type: {JOB_TYPES}
        :type type: str
        :param step: {JOB_STEPS}
        :type step: str
        :param status: Status of the job (in progress, on hold)
        :type status: str
        :param id: Job database identifier
        :type id: UUID
        :param job_arguments: JSON dict containing specific job arguments
        :type job_arguments: dict
        :param createdAt: Time the job was requested
        :type createdAt: datetime
        :param status_details: Additional info about the status
        :type status_details: str
        """

        assert type in JOB_TYPES, f'JobType ({type}) must be in {JOB_TYPES}'
        assert step in JOB_STEPS, f'Job step ({step}) must be in {JOB_STEPS}'
        if type == 'PREDICT':
            assert 'videoId' in job_arguments.keys(), f"VideoId must be specified for predict jobs"
            assert job_arguments['videoId'] is not None, f"VideoId may not be None for predict jobs"
            ValueHelper.check_raise_uuid(job_arguments['videoId'])
            assert 'model' in job_arguments.keys(), f"Model must be specified for predict jobs"
            assert job_arguments['model'] is not None, f"Model may not be None for predict jobs"
            ValueHelper.check_raise_string(job_arguments['model'])

        if type == 'TRAIN':
            assert 'recipe' in job_arguments.keys(), f"Recipe must be specified for predict jobs"
            assert job_arguments['recipe'] is not None, f"Recipe may not be None for predict jobs"
            ValueHelper.check_raise_string(job_arguments['recipe'])

        self.id: UUID = id
        self.type: str = type
        self.step: str = step
        self.job_arguments = job_arguments
        self.createdAt: datetime = createdAt if createdAt else datetime.now()
        self.status: str = status
        self.status_details: str = status_details

    def to_dict(self):
        return {
            'id' : self.id,
            'type' : self.type,
            'step' : self.step,
            'job_arguments' : self.job_arguments,
            'createdAt' : self.createdAt,
            'status' : self.status,
            'status_details' : self.status_details,
        }

