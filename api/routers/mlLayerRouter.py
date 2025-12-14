from flask import request
from flask_restful import Resource
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
    
class MLLayerCompositionMovePropertyRouter(Resource):
    def __init__(self, **kwargs):
        self.mlLayerService = MLLayerService()
        super().__init__(**kwargs)
        
    def post(self):
        data = request.get_json()

        compositionName = data.get('compositionName')
        source_stage = data.get('sourceStage')
        dest_stage = data.get('destStage')
        stageNr = data.get('stageNr')
        key = data.get('key')
            
        try:
            ValueHelper.check_raise_string_only_abc123space(compositionName)
            ValueHelper.check_raise_string_only_abc123space(source_stage)
            ValueHelper.check_raise_string_only_abc123space(dest_stage)
            ValueHelper.check_raise_string_only_abc123space(key)
        
            self.mlLayerService.move_skill_property(
                composition_name=compositionName,
                source=source_stage,
                dest=dest_stage,
                key=key,
                stage=stageNr
            )
        except Exception as e:
            print(e)

        return 'ok', 200 
    
class MLLayerTypesRouter(Resource):
    def __init__(self, **kwargs):
        self.mlLayerService = MLLayerService()
        super().__init__(**kwargs)
    
    def get(self):
        return self.mlLayerService.get_types(), 200

class MLLayerCompositionPropertyAttributeRouter(Resource):
    def __init__(self, **kwargs):
        self.mlLayerService = MLLayerService()
        super().__init__(**kwargs)
        
    def post(self):
        data = request.get_json()

        compositionName = data.get('compositionName')
        stage = data.get('stage')
        propertyname = data.get('propertyname')
        attribute = data.get('attribute')
        value = data.get('value')

        try:
            ValueHelper.check_raise_string_only_abc123space(compositionName)
            ValueHelper.check_raise_string_only_abc123(propertyname)
            ValueHelper.check_raise_string_only_abc123(attribute)
            if stage is not None:
                assert isinstance(stage, int), f"Invalid stage, {stage}"

            self.mlLayerService.update_layer_composition_attribute_value(
                compositionName=compositionName,
                stage=stage,
                attribute=attribute,
                name=propertyname,
                value=value,
            )
        except Exception as e:
            print('MLLayerCompositionPropertyAttributeRouter', e)

        return 'ok', 200 

