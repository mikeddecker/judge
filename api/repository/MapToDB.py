from domain.folder import Folder
from domain.videoinfo import VideoInfo
from repository.models import Folder as FolderDB
from sqlalchemy.orm.session import Session

class MapToDB:

    def map_folder(db, folder: Folder) -> FolderDB:
        session : Session = db.session # For type hint purpose
        return session.get(FolderDB, ident=folder.Id)