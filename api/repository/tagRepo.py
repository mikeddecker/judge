from domain.tag import Tag
from domain.tagGroup import TagGroup
from flask_sqlalchemy import SQLAlchemy
from repository.models import Tag as TagDB
from repository.models import TagGroup as TagGroupDB
from repository.MapToDomain import MapToDomain
from typing import List

class TagRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db
    
    def add(self, name: str, group: str | None) -> Tag:
        if group:
            groupDB = self.db.session.query(TagGroupDB).filter_by(name=group).first()
            new_tag = TagDB(name=name, group=groupDB)
        else:
            new_tag = TagDB(name = name)
        
        self.db.session.add(new_tag)
        self.db.session.commit()
        return MapToDomain.map_tag(new_tag)
    
    def add_group(self, name: str):
        new_tag_group = TagGroupDB(name = name)
        self.db.session.add(new_tag_group)
        self.db.session.commit()
        return MapToDomain.map_tag_group(new_tag_group)
    
    def has_tag(self, id: int) -> bool:
        return self.db.session.query(TagDB).filter_by(id=id).first() is not None
        
    def has_tag_within_group(self, name: str, group: str) -> bool:
        query = self.db.session.query(TagDB).filter_by(name=name)

        if group:
            tag_group = self.db.session.query(TagGroupDB).filter_by(name=group).first()
            if not tag_group:
                return False
            query = query.filter_by(tagGroupId=tag_group.id)
        else:
            query = query.filter_by(tagGroupId=None)

        return self.db.session.query(query.exists()).scalar()
    
    def has_group(self, name: str) -> bool:
        return self.db.session.query(TagGroupDB).filter_by(name=name).first() is not None

    def get_tags(self) -> List[Tag]:
        """
        Returns all tags
        """
        return [MapToDomain.map_tag(t) for t in self.db.session.query(TagDB).all()]

    def get_tag_groups(self) -> List[TagGroup]:
        """
        Returns all tag groups
        """
        return [MapToDomain.map_tag_group(t) for t in self.db.session.query(TagGroupDB).all()]

    def delete(self, id: str) -> None:
        """
        Hard deletes the job from the database.
        """
        if not self.exists(id):
            raise LookupError(f"Folder {id} doesn't exist")
        tagdb = self.db.session.get(TagDB, ident=id)
        self.db.session.delete(tagdb)
        self.db.session.commit()

    def update_tag_group(self, tag_id: int, group_name: str | None):
        tagdb : TagDB = self.db.session.get(TagDB, ident=tag_id)
        if group_name is None:
            tagdb.tagGroupId = None
        else:
            newTagGroup : TagGroupDB = self.db.session.query(TagGroupDB).filter_by(name=group_name).first()
            tagdb.tagGroupId = newTagGroup.id
        self.db.session.commit()

    def update_tag_name(self, tag_id: int, name: str):
        tagdb : TagDB = self.db.session.get(TagDB, ident=tag_id)
        tagdb.name = name
        self.db.session.commit()
