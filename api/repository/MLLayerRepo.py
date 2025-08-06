import json

from domain.layerComposition import LayerComposition
from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, func
from repository.models import LayerProperty, LayerPropertyValue, LayerComposition as LayerCompositionDB
from repository.MapToDomain import MapToDomain
from typing import List

class MLLayerRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db
    
    def add_layer(self, name: str, type:str, min: float = None, max: float = None, step: float = None) -> dict:
        newLayer = LayerProperty(
            name = name,
            type = type,
        )
        if min is not None:
            newLayer.min = min
            newLayer.max = max
            newLayer.step = step

        self.db.session.add(newLayer)
        self.db.session.commit()
        return newLayer.to_dict()
    
    def add_value(self, layerId: int, valueName: str) -> dict:
        """Returns the full layervalue"""
        layer = self.db.session.query(LayerProperty).filter_by(id=layerId).first()
        if layer is None:
            raise ValueError(f"LayerId does not exist ({layerId})")
        if layer.type != 'categorical':
            raise ValueError(f"layer is not categorical")
        
        layerValue = LayerPropertyValue(
            name = valueName, property = layer
        )
        self.db.session.add(layerValue)
        self.db.session.commit()
        return layerValue.to_dict()
    
    def has_layer(self, layerId: int) -> bool:
        return self.db.session.query(LayerProperty).filter_by(id=layerId).scalar() is not None
    
    def has_value(self, layerValueId: int) -> bool:
        return self.db.session.query(LayerPropertyValue).filter_by(id=layerValueId).scalar() is not None
    
    def get(self, layerId: int) -> LayerProperty:
        return self.db.session.get(LayerProperty, ident=layerId)

    def get_all(self) -> dict:
        """
        Returns all layerproperties
        """
        return [lp.to_dict() for lp in self.db.session.query(LayerProperty).order_by(LayerProperty.name).all()]

    def update_layer(self, layerId: int, name: str, min: float = None, max: float = None, step: float = None) -> dict:
        layer = self.db.session.get(LayerProperty, ident=layerId)
        layer.name = name
        layer.min = min
        layer.max = max
        layer.step = step
        layer.lastUpdate = datetime.now

        self.db.session.commit()
        return layer.to_dict()

    def update_value_name(self, layerValueId: int, name: str):
        layervalue : LayerPropertyValue = self.db.session.get(LayerPropertyValue, ident=layerValueId)
        layervalue.name = name
        self.db.session.commit()
        return layervalue.property.to_dict()

    def get_layer_compositions(self) -> dict[str, LayerComposition]:
        compositionValuesDB : list[LayerCompositionDB] = self.db.session.query(
            LayerCompositionDB
        ).join(
            LayerCompositionDB.property
        ).order_by(
            LayerCompositionDB.name,
            LayerProperty.name
        ).all()

        compositions : dict[str, list[LayerCompositionDB]] = dict()
        for comValue in compositionValuesDB:
            if comValue.compositionName in compositions.keys():
                compositions[comValue.compositionName].append(comValue)
            else:
                compositions[comValue.compositionName] = [comValue]
        
        return {compositionName: MapToDomain.map_layercomposition(compositionValues) for compositionName, compositionValues in compositions.items()}

    def add_layer_compostion_stage(self, compositionName: str, stage: int | None, propertyId: int, name: str | None) -> dict[str, LayerComposition]:
        """Return all layer compositions"""
        newLayerCompositionDB = LayerCompositionDB(
            compositionName=compositionName,
            stage=stage,
            propertyId=propertyId,
            name=name
        )

        self.db.session.add(newLayerCompositionDB)
        self.db.session.commit()

        return self.get_layer_compositions()

    # def get_composition(self, compositionName) -> LayerComposition:
    #     compositionValuesDB : list[LayerCompositionDB] = self.db.session.get(
    #         LayerCompositionDB
    #     ).order_by(
    #         LayerCompositionDB.name
    #     ).all()

    #     compositions : dict[str, list[LayerCompositionDB]] = dict()
    #     for comValue in compositionValuesDB:
    #         if comValue.compositionName in compositions.keys():
    #             compositions[comValue.compositionName].add(comValue)
    #         else:
    #             compositions[comValue.compositionName] = [comValue]
        
    #     return {compositionName: MapToDomain.map_layercomposition(compositionValues) for compositionName, compositionValues in compositions.items()}

