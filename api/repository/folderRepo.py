from domain.folder import Folder
from flask_sqlalchemy import SQLAlchemy
from repository.models import Folder as FolderDB
from repository.MapToDomain import MapToDomain
from typing import List

class FolderRepository:
    def __init__(self, db : SQLAlchemy):
        self.db = db
    
    def add(self, name: str, parent: Folder) -> Folder:
        if parent:
            new_folder = FolderDB(name=name, parentId=parent.Id) if parent else Folder(name=name)
        else:
            new_folder = FolderDB(name=name)
        self.db.session.add(new_folder)
        self.db.session.commit()
        return MapToDomain.map_folder(new_folder)
    
    def exists(self, id: int) -> bool:
        if not isinstance(id, int) or id <= 0:
            raise ValueError(f"Id must be a strictly positive integer, got {id}")
        return self.db.session.query(FolderDB.id).filter_by(id=id).scalar() is not None
    
    def exists_by_name(self, name: str, parent: Folder) -> bool:
        # Only when needed
        if not isinstance(name, str) or name.isspace():
            raise ValueError(f"parentId must be a strictly positive integer, got {parent.Id}")
        if parent:
            return self.db.session.query(FolderDB).filter_by(name=name, parentId=parent.Id).scalar() is not None
        else:
            return self.db.session.query(FolderDB).filter_by(name=name).scalar() is not None
        
    def get(self, id: int) -> Folder:
        """
        Query the database and get the folder and all its parents (until its in the root folder, marked NULL in db)

        Args:
            id (int): The ID of the folder to retrieve.

        Returns:
            Folder: The folder object if found.

        Raises:
            LookupError: If no folder with the given ID is found in the database.
            ValueError: If folder_id is not a valid integer (optional validation if desired).
        
        Example:
            >>> folder = get_folder_by_id(1)
            >>> print(folder.path)
            "competition"
        """
        if not self.exists(id):
            raise LookupError(f"Folder {id} doesn't exist")
        folder = self.db.session.get(FolderDB, ident=id)
        return MapToDomain.map_folder(folder)
    
    def get_children(self, id: int) -> List[Folder]:
        if not self.exists(id):
            raise LookupError(f"Folder {id} doesn't exist")
        childrenDB = self.db.session.query(FolderDB).filter_by(parentId=id).all()
        return [MapToDomain.map_folder(c) for c in childrenDB]

    def get_by_name(self, name: str) -> Folder:
        # Only when needed
        raise NotImplementedError

    def delete(self, id):
        """
        Hard deletes the folder from the database.
        """
        if not self.exists(id):
            raise LookupError(f"Folder {id} doesn't exist")
        
        folder = self.db.session.get(FolderDB, ident=id)
        self.db.session.delete(folder)
        self.db.session.commit()
        return True

    def rename(self, id: int, new_name):
        if not self.exists(id):
            raise LookupError(f"Folder {id} doesn't exist")
        
        folder = FolderDB.query.get(id)
        folder.name = new_name
        self.db.session.commit()
        return MapToDomain.map_folder(folder)
    
    def count(self):
        return self.db.session.query(FolderDB).count()
