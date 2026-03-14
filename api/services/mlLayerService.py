from domain.layerComposition import LayerComposition
from repository.db import db
from repository.folderRepo import FolderRepository
from repository.videoRepo import VideoRepository
from repository.jobRepo import JobRepository
from repository.tagRepo import TagRepository
from repository.MLLayerRepo import MLLayerRepository
from helpers.ValueHelper import ValueHelper
from config import LAYER_TYPES
from uuid import UUID

class MLLayerService:
    """Provides the video information of videos"""
    def __init__(self):
        self.VideoRepo = VideoRepository(db=db)
        self.FolderRepo = FolderRepository(db=db)
        self.JobRepo = JobRepository(db=db)
        self.TagRepo = TagRepository(db=db)
        self.MlLayerRepo = MLLayerRepository(db=db)

    def get_layers(self) -> dict:
        """Return account defined layers"""
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

    def add_layer_value(self, layerId: UUID, name: str) -> dict:
        """Returns the full layer"""
        ValueHelper.check_raise_uuid(layerId)
        ValueHelper.check_raise_string_only_abc123space(name)
        if not self.MlLayerRepo.has_layer(layerId):
            raise ValueError(f"Layer with id {layerId} does not exist")
        return self.MlLayerRepo.add_value(layerId=layerId, valueName=name)

    def has_layer(self, id: UUID) -> bool:
        ValueHelper.check_raise_uuid(id)
        return self.MlLayerRepo.has_layer(id)

    def update_layer(self, layerId: UUID, name: str, min: float = None, max: float = None, step: float = None) -> dict:
        """update a layer: can not update type, returns layer"""
        ValueHelper.check_raise_uuid(layerId)
        ValueHelper.check_raise_string_only_abc123space(name)
        assert self.MlLayerRepo.has_layer(layerId), f"Layer ({layerId}) does not exist"

        layer = self.MlLayerRepo.get(layerId)

        if layer.type == 'numerical':
            assert min is not None, f"Min value for numerical values must be filled in"
            assert max is not None, f"Max value for numerical values must be filled in"
            ValueHelper.check_raise_float(min)
            ValueHelper.check_raise_float(max)

            if step is not None:
                ValueHelper.check_raise_float(step)

            return self.MlLayerRepo.update_layer(layerId=layerId, name=name, min=min, max=max, step=step)
        else:
            return self.MlLayerRepo.update_layer(layerId=layerId, name=name)

    def update_value_name(self, layervalueId: UUID, new_name: str):
        ValueHelper.check_raise_uuid(layervalueId)
        ValueHelper.check_raise_string_only_abc123(new_name)
        assert self.MlLayerRepo.has_value(layervalueId), f"Layervalue with id {layervalueId} does not exist"
        return self.MlLayerRepo.update_value_name(layervalueId, new_name)

    def get_layer_compositions(self) -> dict[str, LayerComposition]:
        return self.MlLayerRepo.get_layer_compositions()

    def add_layer_compostion_stage(self, compositionName: str, stage: int | None, layerId: UUID) -> dict[str, LayerComposition]:
        ValueHelper.check_raise_string_only_abc123(compositionName)
        ValueHelper.check_raise_uuid(layerId)
        if stage is not None:
            assert isinstance(stage, int), f"Stage must be an integer"
            assert stage >= -1, f"Stage must be an integer >= -1"

        assert self.MlLayerRepo.has_layer(layerId), f"LayerId {layerId} does not exist"

        return self.MlLayerRepo.add_layer_compostion_stage(
            compositionName=compositionName,
            stage=stage,
            layerId=layerId,
        )

    def update_layer_composition_attribute_value(self, compositionName: str, stage: int | None, attribute:str, name: str, value) -> dict[str, LayerComposition]:
        """Return all layer compositions"""
        ValueHelper.check_raise_string_only_abc123(compositionName)
        ValueHelper.check_raise_string_only_abc123(name)
        if stage is not None:
            assert isinstance(stage, int), f"Stage must be an integer"
            assert stage >= -1, f"Stage must be an integer >= -1"

        assert attribute in ['defaultValue', 'focussed']

        return self.MlLayerRepo.update_layer_compostion_attribute_value(
            compositionName=compositionName,
            stage=stage,
            layername=name,
            attribute=attribute,
            value=value,
        )

    def move_skill_layer(self, composition_name:str, source:str, dest:str, key:str, stage:str|int=None):
        """
        Moves a key-value pair from one section to another within a given composition and entry.

        Parameters:
            composition_name (str): The name of the composition (e.g., 'composition1').
            source (str): The source section: 'GeneralProperties', 'StartProperties', 'EndProperties', or 'StageProperties'.
            dest (str): The destination section. idem source
            key (str): The key to move.
            stage (str|int|None): Required if source or dest is 'StageProperties'.
        """
        assert composition_name is not None
        assert source is not None
        assert dest is not None
        assert key is not None
        assert isinstance(composition_name, str)
        assert isinstance(source, str)
        assert isinstance(dest, str)
        assert isinstance(key, str)

        # TODO : assert exists

        self.MlLayerRepo.move_skill_layer(composition_name=composition_name, source=source, dest=dest, key=key, stage=stage)

