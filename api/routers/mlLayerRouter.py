from flask import request, Response
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from services.jobService import JobService
from services.storageService import StorageService
from services.tagService import TagService
from services.mlLayerService import MLLayerService
from helpers.ValueHelper import ValueHelper
from config import LAYER_TYPES

class MLLayerRouter(Resource):
    def __init__(self, **kwargs):
        self.mlLayerService = MLLayerService()
        super().__init__(**kwargs)
    
    def get(self):
        return self.mlLayerService.get_layers(), 200
    
    def post(self):
        data = request.get_json()
        name = data.get('name')
        layerId = data.get('layerId')
        type = data.get('type')
        
        ValueHelper.check_raise_string_only_abc123space(name)
        if layerId is None:
            # Layer
            assert type in LAYER_TYPES, f"Type does not exist"

            added = self.mlLayerService.add_layer(
                name = name,
                type = type,
                min = data.get('min'),
                max = data.get('max'),
                step = data.get('step'),
            )
        else:
            # Value
            ValueHelper.check_raise_id(layerId)
            assert self.mlLayerService.has_layer(layerId), f"Layer does not exist"
            added = self.mlLayerService.add_layer_value(layerId=layerId, name=name)
        return added, 200
    
    def put(self):
        data = request.get_json()
        id = data.get('id')
        name = data.get('name')
        layerId = data.get('layerId')
        
        ValueHelper.check_raise_string_only_abc123(name)
        if id is not None:
            # Layer
            return self.mlLayerService.update_layer(
                layerId = id,
                name = name,
                min = data.get('min'),
                max = data.get('max'),
                step = data.get('step'),
            ), 200
        elif layerId is not None:
            # Value
            ValueHelper.check_raise_id(layerId)
            assert self.mlLayerService.has_layer(layerId), f"Layer does not exist"
            return self.mlLayerService.update_value_name(layervalueId=layerId, new_name=name), 200
        else:
            return 'Invalid layer (value) update', 404
    
class MLLayerCompositionRouter(Resource):
    def __init__(self, **kwargs):
        self.mlLayerService = MLLayerService()
        super().__init__(**kwargs)
    
    def get(self):
        return {
            compositionName: layerComposition.to_dict()
            for compositionName, layerComposition in
            self.mlLayerService.get_layer_compositions().items()
        }, 200
    
    def post(self):
        data = request.get_json()

        compositionName = data.get('compositionName')
        stage = data.get('stage')
        propertyId = data.get('propertyId')
        name = data.get('name')
        
        ValueHelper.check_raise_string_only_abc123(compositionName)
        ValueHelper.check_raise_id(propertyId)
        if stage is not None:
            assert isinstance(stage, int), f"Stage must be an integer"
            assert stage >= -1, f"Stage must be an integer >= -1"
        if name is not None:
            ValueHelper.check_raise_string_only_abc123(name)
        
        assert self.mlLayerService.has_layer(propertyId), f"LayerPropertyId {propertyId} does not exist"

        return {
            compositionName: layerComposition.to_dict()
            for compositionName, layerComposition in
            self.mlLayerService.add_layer_compostion_stage(
                compositionName=compositionName,
                stage=stage,
                propertyId=propertyId,
                name=name
            ).items()
        }, 200 
    
    # def put(self):
    #     data = request.get_json()
    #     id = data.get('id')
    #     name = data.get('name')
    #     layerId = data.get('layerId')
        
    #     ValueHelper.check_raise_string_only_abc123(name)
    #     if id is not None:
    #         # Layer
    #         return self.mlLayerService.update_layer(
    #             layerId = id,
    #             name = name,
    #             min = data.get('min'),
    #             max = data.get('max'),
    #             step = data.get('step'),
    #         ), 200
    #     elif layerId is not None:
    #         # Value
    #         ValueHelper.check_raise_id(layerId)
    #         assert self.mlLayerService.has_layer(layerId), f"Layer does not exist"
    #         return self.mlLayerService.update_value_name(layervalueId=layerId, new_name=name), 200
    #     else:
    #         return 'Invalid layer (value) update', 404

class MLLayerTypesRouter(Resource):
    def __init__(self, **kwargs):
        self.mlLayerService = MLLayerService()
        super().__init__(**kwargs)
    
    def get(self):
        return self.mlLayerService.get_types(), 200

# class MLLayerValueRouter(Resource):
#     def __init__(self, **kwargs):
#         self.folderService = FolderService()
#         self.videoService = VideoService()
#         self.jobService = JobService()
#         self.tagService = TagService()
#         super().__init__(**kwargs)
    
#     def get(self):
#         return [t.to_dict() for t in self.tagService.get_tag_groups()], 200

#     def post(self):
#         data = request.get_json()
#         name = data.get('name')
#         ValueHelper.check_raise_string_only_abc123(name)
#         return self.tagService.add_group(name).to_dict(), 200

