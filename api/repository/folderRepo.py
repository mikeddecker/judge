from domain.folder import Folder
from models import db
from typing import List

class folderRepository:
    def add(name : str) -> Folder:
        pass

    def get(id : int) -> Folder:
        pass
    
    def get_children_from(parent : str) -> List[Folder]:
        pass

    def delete(id) -> None:
        pass

    def rename(id : int, name : str) -> None:
        pass
