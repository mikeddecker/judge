# Does in depth checks
import os
from domain.folder import Folder
from repository.db import db
from repository.folderRepo import FolderRepository
from services.ServiceHelper import ServiceHelper

class FolderService:
    def __init__(self, storage_folder: str):
        ServiceHelper.check_raise_string(storage_folder)
        self.folderRepo = FolderRepository(db=db)
        self.storage_folder = storage_folder
        if not self.exists_path_on_drive(self.storage_folder):
            raise ValueError(f"folder {storage_folder} does not exist")

    def add_in_database(self, name, parent: Folder) -> Folder:
        """
        Adds an existing folder on the storage_drive to the database.
        """
        if self.exists_in_database(name=name, parent=parent):
            raise LookupError(f"{name} found in db: {parent.get_relative_path()}")
        if not self.exists_path_on_drive(name=name, parent=parent):
            raise ValueError(f"folder {name} not found in {self.storage_folder if not parent else os.path.join(self.storage_folder, parent.get_relative_path())}")
        return self.folderRepo.add(name=name, parent=parent)
    
    def create_on_drive(self, name, parent: Folder):
        """
        Effectively creates the folder on the drive

        :raises: NotADirectoryError - Parent is not a (nested) directory
        """
        ServiceHelper.check_raise_string(name)
        if parent:
            # Parent folder exists?
            print(parent.get_relative_path())
            if self.exists_path_on_drive(parent.Name, parent.Parent):
                os.mkdir(os.path.join(self.storage_folder, parent.get_relative_path(), name))
            else:
                raise NotADirectoryError(f"Parent is not a (nested) directory, got full path is : {os.path.join(self.storage_folder, parent.get_relative_path())}")
        else:
            os.mkdir(os.path.join(self.storage_folder, name))

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
            ServiceHelper.check_raise_id(id)
            return self.folderRepo.exists_by_id(id)
        if name:
            ServiceHelper.check_raise_string(name)
            return self.folderRepo.exists_by_name(name=name, parent=parent)

    def exists_path_on_drive(self, name: str, parent: Folder = None) -> bool:
        if parent:
            print("Name is ", name, "parent is", parent.get_relative_path())
            return os.path.exists(os.path.join(self.storage_folder, parent.get_relative_path(), name))
        return os.path.exists(os.path.join(self.storage_folder, name))

    def get(self, id: int):
        ServiceHelper.check_raise_id(id)
        self.folderRepo.get(id)

    def rename(self, id: int, new_name: str):
        ServiceHelper.check_raise_id(id)
        ServiceHelper.check_raise_string(new_name)
        if not self.exists_in_database(id=id):
            raise LookupError(f"Folder {id} not found in db")
        self.folderRepo.rename(id=id, new_name=new_name)

    def delete(self, id: int):
        ServiceHelper.check_raise_id(id)
        if not self.exists_in_database(id=id):
            raise LookupError(f"Folder {id} not found in db")
        self.folderRepo.delete(id=id)

    def count(self):
        return self.folderRepo.count()