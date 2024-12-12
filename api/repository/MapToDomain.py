from domain.folder import Folder
from domain.videoinfo import VideoInfo
from repository.models import Folder as FolderDB

class MapToDomain:
    def map_folder(folderDB: FolderDB) -> Folder:
        folder = Folder(folderDB.id, folderDB.name, None)
        while folderDB.parent:
            parent_folder = Folder(folderDB.parent.id, folderDB.parent.name, None)
            folder.Parent = parent_folder
            folderDB = folderDB.parent
        return folder