from models import db, Folder as FolderDB
from domain.folder import Folder

class MapToDomain:
    def map_folder(folderDB: FolderDB) -> Folder:
        folder = Folder(folderDB.id, folderDB.name, None)
        while folderDB.parent:
            parent_folder = Folder(folderDB.parent.id, folderDB.parent.name, None)
            folder.Parent = parent_folder
            folderDB = folderDB.parent