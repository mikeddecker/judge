from flask_restful import Resource
from flask import request
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper
from uuid import UUID

class FolderRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService()
        self.videoService = VideoService()
        super().__init__(**kwargs)
    
    def get(self, folderId: UUID = None):
        if folderId:
            try:
                if not self.folderService.exists_in_database(id=folderId):
                    return f"FolderId {folderId} does not exist", 404
            except ValueError as ve:
                print(ve)
                return f'Something went wrong at the FolderRouter', 500
            f = self.folderService.get(folderId).to_dict()
            children = self.folderService.get_children(f["Id"])
            f["Children"] = []
            for c in children:
                f["Children"].append(c.to_dict())
            f["Videos"] = {}
            for vidinfo in self.videoService.get_videos(folderId=f["Id"]):
                f["Videos"][str(vidinfo.Id)] = vidinfo.to_dict(include_frames=False)
            f["VideoCount"] = len(f["Videos"].keys())
            return f, 200
        else:
            # Modify to represent the same output as get(folderId)
            return {
                "Id" : 0,
                "Name" : "root",
                "Children" : [f.to_dict() for f in self.folderService.get_root_folders()],
                "Parent" : None,
                "Videos" : dict(),
                "VideoCount" : 0,
            }, 200

    def post(self, folderId: UUID):
        data = request.get_json()
        
        print('folderrouter post', data)
        
        # Validate UUID
        try:
            ValueHelper.check_raise_uuid(folderId)
        except ValueError as ve:
            return {'success': False, 'message': str(ve)}, 400
        
        # Allow updating training status without authentication
        # This supports changing folder train/val assignment via GUI
        updated_folder = self.folderService.update_folder_no_auth(
            folderId=folderId,
            updatedData=data
        )
        
        # return serialized domain object
        return updated_folder.to_dict(), 200

