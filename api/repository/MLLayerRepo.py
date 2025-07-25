import json

from datetime import datetime, date
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import and_, func
from repository.models import LayerProperty, LayerPropertyValue
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
    
    def get_all(self) -> dict:
        """
        Returns all jobs
        """
        return [lp.to_dict() for lp in self.db.session.query(LayerProperty).all()]

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

    # def delete(self, id: int) -> None:
    #     """
    #     Hard deletes the job from the database.
    #     """
    #     if not self.exists(id):
    #         raise LookupError(f"Folder {id} doesn't exist")
    #     jobdb = self.db.session.get(JobDB, ident=id)
    #     self.db.session.delete(jobdb)
    #     self.db.session.commit()


    # def count(self) -> int:
    #     return self.db.session.query(JobDB).count()
