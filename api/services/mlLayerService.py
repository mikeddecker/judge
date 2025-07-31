from domain.tag import Tag
from domain.tagGroup import TagGroup
from domain.layerComposition import LayerComposition
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.jobRepo import JobRepository
from repository.tagRepo import TagRepository
from repository.MLLayerRepo import MLLayerRepository
from typing import List
from helpers.ValueHelper import ValueHelper
from config import LAYER_TYPES
from pprint import pprint

class MLLayerService:
    """Provides the video information of videos"""
    def __init__(self):
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)
        self.JobRepo = JobRepository(db=db)
        self.TagRepo = TagRepository(db=db)
        self.MlLayerRepo = MLLayerRepository(db=db)
        
    def get_layers(self) -> dict:
        """Return user defined layers"""
        return self.MlLayerRepo.get_all()

    def get_types(self) -> list:
        """Return layers types"""
        return LAYER_TYPES

    def add_layer(self, name: str, type:str, min: float = None, max: float = None, step: float = None) -> dict:
        """Adding a layer: categorical, numeric or boolean"""
        
        ValueHelper.check_raise_string_only_abc123(name)
        ValueHelper.check_raise_string_only_abc123(type)
        assert type in LAYER_TYPES, f"Unknown type {type}"

        if type == 'numerical':
            assert min is not None, f"Min value for numerical values must be filled in"
            assert max is not None, f"Max value for numerical values must be filled in"
            ValueHelper.check_raise_float(min)
            ValueHelper.check_raise_float(max)

            if step is not None:
                ValueHelper.check_raise_float(step)

            return self.MlLayerRepo.add_layer(name=name, type=type, min=min, max=max, step=step)
        else:
            return self.MlLayerRepo.add_layer(name=name, type=type)
    
    def add_layer_value(self, layerId: int, name: str) -> dict:
        """Returns the full layer"""
        ValueHelper.check_raise_id(layerId)
        ValueHelper.check_raise_string_only_abc123(name)
        if not self.MlLayerRepo.has_layer(layerId):
            raise ValueError(f"Layer with id {layerId} does not exist")
        return self.MlLayerRepo.add_value(layerId=layerId, valueName=name)

    def has_layer(self, id: int) -> bool:
        ValueHelper.check_raise_id(id)
        return self.MlLayerRepo.has_layer(id)
    
    def update_layer(self, layerId: int, name: str, min: float = None, max: float = None, step: float = None) -> dict:
        """update a layer: can not update type, returns layer"""
        ValueHelper.check_raise_id(layerId)
        ValueHelper.check_raise_string_only_abc123(name)
        ValueHelper.check_raise_string_only_abc123(type)
        assert type in LAYER_TYPES, f"Unknown type {type}"
        assert self.MlLayerRepo.has_layer(layerId), f"Layer ({layerId}) does not exist"

        if type == 'numerical':
            assert min is not None, f"Min value for numerical values must be filled in"
            assert max is not None, f"Max value for numerical values must be filled in"
            ValueHelper.check_raise_float(min)
            ValueHelper.check_raise_float(max)

            if step is not None:
                ValueHelper.check_raise_float(step)

            return self.MlLayerRepo.update_layer(layerId=layerId, name=name, min=min, max=max, step=step)
        else:
            return self.MlLayerRepo.update_layer(layerId=layerId, name=name)
    
    def update_value_name(self, layervalueId: int, new_name: str):
        ValueHelper.check_raise_id(layervalueId)
        ValueHelper.check_raise_string_only_abc123(new_name)
        assert self.MlLayerRepo.has_value(layervalueId), f"Layervalue with id {layervalueId} does not exist"
        return self.MlLayerRepo.update_value_name(layervalueId, new_name)

    def get_layer_compositions(self) -> dict[str, LayerComposition]:
        return self.MlLayerRepo.get_layer_compositions()
    
    def add_layer_compostion_stage(self, compositionName: str, stage: int | None, propertyId: int, name: str | None) -> dict[str, LayerComposition]:
        ValueHelper.check_raise_string_only_abc123(compositionName)
        ValueHelper.check_raise_id(propertyId)
        if stage is not None:
            assert isinstance(stage, int), f"Stage must be an integer"
            assert stage >= -1, f"Stage must be an integer >= -1"
        if name is not None:
            ValueHelper.check_raise_string_only_abc123(name)
        
        assert self.MlLayerRepo.has_layer(propertyId), f"LayerPropertyId {propertyId} does not exist"
        
        return self.MlLayerRepo.add_layer_compostion_stage(
            compositionName=compositionName, 
            stage=stage, 
            propertyId=propertyId, 
            name=name
        )

