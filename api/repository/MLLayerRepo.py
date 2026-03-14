
from datetime import datetime
from uuid import UUID
from domain.layerComposition import LayerComposition
from flask_sqlalchemy import SQLAlchemy
from repository.models import Layer, LayerValue, LayerComposition as LayerCompositionDB, Skill as SkillDB
from repository.MapToDomain import MapToDomain
from sqlalchemy import or_
from sqlalchemy.orm.attributes import flag_modified

class MLLayerRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db
    
    def add_layer(self, name: str, type:str, min: float = None, max: float = None, step: float = None) -> dict:
        newLayer = Layer(
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
    
    def add_value(self, layerId: UUID, valueName: str) -> dict:
        """Returns the full layervalue"""
        layer = self.db.session.query(Layer).filter_by(id=layerId).first()
        if layer is None:
            raise ValueError(f"LayerId does not exist ({layerId})")
        if layer.type != 'categorical':
            raise ValueError(f"layer is not categorical")
        
        layerValue = LayerValue(
            name = valueName, layer = layer
        )
        self.db.session.add(layerValue)
        self.db.session.commit()
        return layerValue.to_dict()
    
    def has_layer(self, layerId: UUID) -> bool:
        return self.db.session.query(Layer).filter_by(id=layerId).scalar() is not None
    
    def has_value(self, layerValueId: UUID) -> bool:
        return self.db.session.query(LayerValue).filter_by(id=layerValueId).scalar() is not None
    
    def get(self, layerId: UUID) -> Layer:
        return self.db.session.get(Layer, ident=layerId)

    def get_all(self) -> dict:
        """
        Returns all layers
        """
        return [lp.to_dict() for lp in self.db.session.query(Layer).order_by(Layer.name).all()]

    def update_layer(self, layerId: UUID, name: str, min: float = None, max: float = None, step: float = None) -> dict:
        layer = self.db.session.get(Layer, ident=layerId)
        layer.name = name
        layer.min = min
        layer.max = max
        layer.step = step
        layer.lastUpdate = datetime.now

        self.db.session.commit()
        return layer.to_dict()

    def update_value_name(self, layerValueId: UUID, name: str):
        layervalue : LayerValue = self.db.session.get(LayerValue, ident=layerValueId)
        layervalue.name = name
        self.db.session.commit()
        return layervalue.layer.to_dict()

    def get_layer_compositions(self) -> dict[str, LayerComposition]:
        compositionValuesDB : list[LayerCompositionDB] = self.db.session.query(
            LayerCompositionDB
        ).join(
            LayerCompositionDB.layer
        ).order_by(
            LayerCompositionDB.compositionName,
            Layer.name
        ).all()

        compositions : dict[str, list[LayerCompositionDB]] = dict()
        for comValue in compositionValuesDB:
            if comValue.compositionName in compositions.keys():
                compositions[comValue.compositionName].append(comValue)
            else:
                compositions[comValue.compositionName] = [comValue]
        
        return {compositionName: MapToDomain.map_layercomposition(compositionValues) for compositionName, compositionValues in compositions.items()}

    def add_layer_compostion_stage(self, compositionName: str, stage: int | None, layerId: UUID) -> dict[str, LayerComposition]:
        """Return all layer compositions"""
        newLayerCompositionDB = LayerCompositionDB(
            compositionName=compositionName,
            stage=stage,
            layerId=layerId,
        )

        self.db.session.add(newLayerCompositionDB)
        self.db.session.commit()

        return self.get_layer_compositions()
    
    def update_layer_compostion_attribute_value(self, compositionName: str,  stage: int, layername: str, attribute: str, value):
        layer_composition_row: LayerCompositionDB = self.db.session.query(
            LayerCompositionDB
        ).join(
            LayerCompositionDB.layer
        ).filter(
            LayerCompositionDB.compositionName==compositionName,
            LayerCompositionDB.stage==stage,
            or_(
                LayerCompositionDB.name==layername,
                Layer.name==layername,
            )
        ).first()

        if layer_composition_row is None:
            raise ValueError(f"{compositionName} - {stage} - {layername} does not exist")
        
        match attribute:
            case 'defaultValue':
                layer_composition_row.defaultValue = value
            case 'focussed':
                layer_composition_row.focussed = value
            case _:
                print('?', attribute)
        self.db.session.commit()

        return self.get_layer_compositions()

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
        assert self.db.session.query(LayerCompositionDB).filter_by(compositionName=composition_name).first() is not None
        assert source in {'GeneralProperties', 'StartProperties', 'EndProperties', 'StageProperties'}
        assert dest in {'GeneralProperties', 'StartProperties', 'EndProperties', 'StageProperties'}

        movingTime = datetime.now()
        try:
            skills : list[SkillDB] = self.db.session.query(SkillDB).all()
            for skillDB in skills:
                if composition_name not in skillDB.skillinfo.keys():
                    continue

                for composition_entry_index in range(len(skillDB.skillinfo[composition_name])):
                    # Determine source dictionary
                    if source == "StageProperties":
                        if stage is None:
                            raise ValueError("Stage number required for StageProperties source.")
                        if key not in skillDB.skillinfo[composition_name][composition_entry_index]["StageProperties"][stage].keys():
                            continue

                        layervalue = skillDB.skillinfo[composition_name][composition_entry_index]["StageProperties"][stage].pop(key)
                    else:
                        if key not in skillDB.skillinfo[composition_name][composition_entry_index][source].keys():
                            continue
                        layervalue = skillDB.skillinfo[composition_name][composition_entry_index][source].pop(key)

                    # Determine destination dictionary
                    if dest == "StageProperties":
                        if stage is None:
                            raise ValueError("Stage number required for StageProperties destination.")
                        skillDB.skillinfo[composition_name][composition_entry_index]["StageProperties"].setdefault(stage, {})
                        skillDB.skillinfo[composition_name][composition_entry_index]["StageProperties"][stage][key] = layervalue
                    else:
                        skillDB.skillinfo[composition_name][composition_entry_index][dest][key] = layervalue

                skillDB.skillinfo = dict(skillDB.skillinfo)
                skillDB.updated = movingTime
                flag_modified(skillDB, "skillinfo")
                self.db.session.flush()
                self.db.session.merge(skillDB)
                self.db.session.commit()

            try:
                source_stage_nr = STAGE_MAP[source] if stage is None else stage
                dest_stage_nr = STAGE_MAP[dest] if stage is None else stage
                layer_composition_part: LayerCompositionDB = self.db.session.query(LayerCompositionDB).filter_by(compositionName=composition_name, stage=source_stage_nr, name=key).first()
                if layer_composition_part is None:
                    # Meaning, no custom name
                    layerprop : Layer = self.db.session.query(Layer).filter_by(name=key).first()
                    if layerprop is None:
                        raise ValueError(f"Layer (Key) {key} does not exist")
                    layer_composition_part: LayerCompositionDB = self.db.session.query(LayerCompositionDB).filter_by(compositionName=composition_name, stage=source_stage_nr, layerId=layerprop.id).first()
                    if layer_composition_part is None:
                        raise ValueError(f"Unknown error: could not find composition {composition_name} - stage {source_stage_nr} - layer (key={key} - id={layerprop.id})")

                layer_composition_part.stage = dest_stage_nr
                
                self.db.session.commit()
            except Exception as e:
                self.db.session.rollback()
                print(str(e))

        except Exception as e:
            skill_id = skillDB.id if 'skillDB' in locals() else 'Unknown'
            print(f"Error moving key '{key}': {e} in skillDB with Id {skill_id}")
            self.db.session.rollback()

