# Does in depth checks
import os
from domain.folder import Folder
from repository.db import db
from repository.folderRepo import FolderRepository
from helpers.ValueHelper import ValueHelper

class FolderService:
    PROPERTIES = [
        "FolderRepo",
        "StorageFolder",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        self.FolderRepo = FolderRepository(db=db)

        if not os.path.exists(storage_folder):
            raise NotADirectoryError(f"StorageFolder {storage_folder} does not exist")
        self.StorageFolder = storage_folder
        
    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Prevent setting immutable attributes after it is set in __init__
            if name in self.PROPERTIES:
                raise AttributeError(f"Cannot modify {name} once it's set")
        elif name not in self.PROPERTIES:
            raise NameError(f"Property {name} does not exist")
        super().__setattr__(name, value)

    def add_in_database(self, name, parent: Folder) -> Folder:
        """
        Adds an existing folder on the storage_drive to the database.
        """
        ValueHelper.check_raise_string_only_abc123(name)

        # Parent exists?
        nested_parent = parent
        while nested_parent:
            if not self.exists_in_database(id=nested_parent.Id, name=nested_parent.Name, parent=nested_parent.Parent):
                raise LookupError(f"(Nested) parent does not exist: {nested_parent}")
            nested_parent = nested_parent.Parent

        # Do I exist?
        if self.exists_in_database(name=name, parent=parent):
            raise LookupError(f"{name} found in db: {parent.get_relative_path()}")
        if not self.exists_path_on_drive(name=name, parent=parent):
            raise ValueError(f"folder {name} not found in {self.StorageFolder if not parent else os.path.join(self.StorageFolder, parent.get_relative_path())}")
        return self.FolderRepo.add(name=name, parent=parent)
    
    def create_on_drive(self, name, parent: Folder = None):
        """
        Effectively creates the folder on the drive

        :raises: NotADirectoryError - Parent is not a (nested) directory
        """
        ValueHelper.check_raise_string(name)
        # Already exists?
        if parent and os.path.exists(os.path.join(self.StorageFolder, parent.get_relative_path(), name)):
            raise FileExistsError(f"folder already exists: {os.path.join(self.StorageFolder, parent.get_relative_path(), name)}")
        elif os.path.exists(os.path.join(self.StorageFolder, name)):
             raise FileExistsError(f"folder already exists: {os.path.join(self.StorageFolder, name)}")

        # Non existent
        if parent:
            # Parent folder exists?
            if self.exists_in_database(name=name, parent=parent):
                raise LookupError(f"{name} found in db: {parent.get_relative_path()}")
            if not self.exists_path_on_drive(parent.Name, parent.Parent):
                raise NotADirectoryError(f"Parent is not a (nested) directory, got full path is : {os.path.join(self.StorageFolder, parent.get_relative_path())}")
            os.mkdir(os.path.join(self.StorageFolder, parent.get_relative_path(), name))
        else:
            os.mkdir(os.path.join(self.StorageFolder, name))

        return self.add_in_database(name=name, parent=parent)

    def exists_in_database(self, id: int = None, name: str = None, parent: Folder = None) -> bool:
        """
        Query db to know if a folder exists.
        If id specified, ignore name and parent
        If no id specified, use name and parent

        Raises:
            ValueError error when id or name are empty or smaller then 0
        """
        if id:
            ValueHelper.check_raise_id(id)
            return self.FolderRepo.exists(id)
        else:
            ValueHelper.check_raise_string_only_abc123(name)
            return self.FolderRepo.exists_by_name(name=name, parent=parent)

    def exists_path_on_drive(self, name: str, parent: Folder = None) -> bool:
        if parent:
            return os.path.exists(os.path.join(self.StorageFolder, parent.get_relative_path(), name))
        return os.path.exists(os.path.join(self.StorageFolder, name))

    def get(self, id: int):
        ValueHelper.check_raise_id(id)
        self.FolderRepo.get(id)

    def rename(self, id: int, new_name: str):
        ValueHelper.check_raise_id(id)
        ValueHelper.check_raise_string(new_name)
        if not self.exists_in_database(id=id):
            raise LookupError(f"Folder {id} not found in db")
        self.FolderRepo.rename(id=id, new_name=new_name)

    def delete(self, id: int):
        ValueHelper.check_raise_id(id)
        if not self.exists_in_database(id=id):
            raise LookupError(f"Folder {id} not found in db")
        self.FolderRepo.delete(id=id)

    def count(self):
        return self.FolderRepo.count()