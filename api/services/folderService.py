# Does in depth checks
import os
from domain.folder import Folder
from repository.folderRepo import FolderRepository
from ServiceHelper import ServiceHelper

STORAGE_DIR = os.getenv("STORAGE_DIR")
class FolderManager:
    def __init__(self):
        self.folderRepo = FolderRepository()


    def add_in_database(self, name, parent: Folder) -> Folder:
        """
        Adds an existing folder on the storage_drive to the database.
        """
        if self.exists_in_db(name=name, parent=parent):
            raise LookupError(f"{name} found in db: {parent.get_relative_path()}")
        if self.exists_path(name=name, parent=parent):
            raise ValueError(f"folder {name} not found in {parent.get_relative_path()}")
        return self.folderRepo.add(name=name, parent=parent)

    def exists_in_db(self, id: int = None, name: str = None, parent: Folder = None) -> bool:
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
            return self.folderRepo.exists_name(name=name, parent=parent)

    def exists_path(self, name: str = None, parent: Folder = None) -> bool:
        if parent:
            return os.path.exists(os.path.join(STORAGE_DIR, name, parent.get_relative_path()))
        return os.path.exists(os.path.join(STORAGE_DIR, name))

    def get(self, id: int):
        ServiceHelper.check_raise_id(id)
        self.folderRepo.get(id)

    def rename(self, id: int, new_name: str):
        ServiceHelper.check_raise_id(id)
        ServiceHelper.check_raise_string(new_name)
        if not self.exists_in_db(id=id):
            raise LookupError(f"Folder {id} not found in db")
        self.folderRepo.rename(id=id, new_name=new_name)

    def delete(self, id: int):
        ServiceHelper.check_raise_id(id)
        if not self.exists_in_db(id=id):
            raise LookupError(f"Folder {id} not found in db")
        self.folderRepo.delete(id=id)