# Browse and navigates storage device, to find and discover videos.

# Browse and navigate storage to find DB orphans

# Nice to have : Browse and navigate storage to find renames that happend on the drive

## Legacy code

    # # Read from the most inner nested to outside
    # return list(
    #     map(
    #         lambda vid : {
    #             "title" : vid,
    #             "image-preview" : os.path.join(API_URL, "videoimage", subFolder, urllib.parse.quote(vid))
    #         },
    #         # Map file string title to
    #         filter(
    #             # Filter supported formats e.g. ".mp4"
    #             lambda f : f.split('.')[-1].lower() in SUPPORTED_VIDEO_FORMATS,
    #             filter(
    #                 # Filter files from files and subfolders
    #                 lambda content : os.path.isfile(os.path.join(VIDEO_FOLDER, subFolder, content)), 
    #                 os.listdir(os.path.join(VIDEO_FOLDER, subFolder))
    #             )
    #         )
    #     )
    # )
import os
import time
import traceback

from colorama import Fore, Style
from domain.folder import Folder
from domain.videoinfo import VideoInfo
from domain.frameinfo import FrameInfo
from helpers.ValueHelper import ValueHelper
from services.videoService import VideoService
from services.folderService import FolderService
from repository.db import db
from typing import List
from dotenv import load_dotenv

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
SUPPORTED_IMAGE_FORMATS = os.getenv("SUPPORTED_IMAGE_FORMATS")
VIDEO_IMAGE_PREVIEW_FOLDER = os.getenv("VIDEO_IMAGE_PREVIEW_FOLDER")

# pseudo cache
cache = {
    "result": None,
    "timestamp": None
}

class StorageService:
    """Provides the video information of videos"""
    PROPERTIES = [
        "VideoRepo",
        "FolderRepo",
        "StorageFolder",
    ]
    def __init__(self, storage_folder: str):
        ValueHelper.check_raise_string(storage_folder)
        if not os.path.exists(storage_folder):
            raise NotADirectoryError(f"StorageFolder {storage_folder} does not exist")
        self.VideoService = VideoService(STORAGE_DIR)
        self.FolderService = FolderService(STORAGE_DIR)

    def discover_drive_cached_version(self, deleteOrphans: bool = False):
        """Pseudo cache method"""
        current_time = time.time()
        WAIT_TIME = 60
        
        if cache["timestamp"] is None or (current_time - cache["timestamp"] > 60):
            new_and_orphans = self.__discover_drive(deleteOrphans=deleteOrphans)
            cache["result"] = new_and_orphans
            finish_time = time.time()
            cache["timestamp"] = finish_time
            new_and_orphans["timestamp"] = finish_time
            new_and_orphans["remaining"] = WAIT_TIME
            new_and_orphans["orphans"]["deleted"] = deleteOrphans
            return new_and_orphans
        
        # If cache is still valid, return the cached result
        time_remaining = WAIT_TIME - (current_time - cache["timestamp"])
        cached_result = cache["result"]
        cached_result["remaining"] = time_remaining
        return cached_result

    def __discover_drive(self, deleteOrphans: bool = False) -> dict:
        try:    
            print(f"{Fore.YELLOW}Discovering folder:{Style.RESET_ALL}", f"{STORAGE_DIR} (root)")
            new_videos, orphans = self.__discover_folder(STORAGE_DIR, parent=None, isRoot=True, deleteOrphans=deleteOrphans)
            return {
                "metadata" : {
                    "new-videos" : "folderId -> name",
                    "orpahs" : "folderId -> videoId -> name"
                },
                "new-videos" : new_videos,
                "orphans" : orphans
            }
        except Exception as e:
            print(traceback.format_exc())
            print(e)

    def __discover_folder(self, currentFolder: str, parent: Folder, isRoot=False, deleteOrphans=False):
        if currentFolder is None or not isinstance(currentFolder, str):
            raise ValueError(f"Didn't get a string for folder, got", currentFolder)
        if not isRoot and (parent is None or not isinstance(parent, Folder)):
            raise ValueError(f"Not in root folder, but got no parent, got {parent} instead")
        
        currentFolderPath = currentFolder if isRoot else os.path.join(STORAGE_DIR, parent.get_relative_path())
        folder_content = os.listdir(currentFolderPath)
        videos_in_folder_according_to_database = {} if isRoot else self.VideoService.get_videos(parent.Id)
        videos_in_folder_according_to_database = { videoinfo.Name : videoinfo for videoinfo in videos_in_folder_according_to_database }
        children = []
        new_videos = {}
        orphans = {}
        new_videos_name_key = 'new-videos'

        for content in folder_content:
            contentPath = os.path.join(currentFolderPath, content)
            
            # Rename files with spaces
            if content.__contains__(" "):
                old_name = contentPath
                contentPath = contentPath.replace(" ", "-")
                content = content.replace(" ", "-")
                os.rename(old_name, contentPath)
                print(f"{Fore.MAGENTA}File or folder contains spaces, renamed with (-) dashes:{Style.RESET_ALL}", content)
            
            # Temp save dirs, to provide better output
            # Otherwise videos, and folders interlap with each other
            if os.path.isdir(contentPath):
                children.append({"name": content, "parent": parent})

            # Handle files
            if os.path.isfile(contentPath):
                if isRoot:
                    print(f"{Fore.YELLOW}Skipping file in root:{Style.RESET_ALL} {content}")
                elif content.split(".")[-1] in SUPPORTED_VIDEO_FORMATS:
                    is_new = ""
                    if self.VideoService.exists_in_database(name=content, folder=parent):
                        del videos_in_folder_according_to_database[content]
                    else:
                        self.VideoService.add(name=content, folder=parent, frameLength=222)
                        # Bookkeeping
                        if parent.Id in new_videos.keys():
                            new_videos[parent.Id].append(content)
                        else:
                            new_videos[parent.Id] = [content]
                        is_new = f"{Fore.GREEN}NEW{Style.RESET_ALL}"
                    print(f"{Fore.LIGHTBLUE_EX}Detected video: {Style.RESET_ALL} {content}", is_new)
                elif content.split(".")[-1] in SUPPORTED_IMAGE_FORMATS:
                    print(f"{Fore.LIGHTMAGENTA_EX}Detected image:{Style.RESET_ALL} {content} (currently skipped)")
                else:
                    print(f"{Fore.YELLOW}Detected other:{Style.RESET_ALL} {content}")

        for orpan_name, videoinfo in videos_in_folder_according_to_database.items():
            print(f"{Fore.RED}Detected orphan: videoId = {Fore.YELLOW}{videoinfo.Id}{Style.RESET_ALL} {orpan_name}")
            if deleteOrphans:
                self.VideoService.delete_from_database(id=videoinfo.Id)
            if parent.Id in orphans.keys():
                orphans[parent.Id][videoinfo.Id] = orpan_name
            else:
                orphans[parent.Id] = { videoinfo.Id : orpan_name }

        # Now loop al children
        reserved_names = [VIDEO_IMAGE_PREVIEW_FOLDER, new_videos_name_key]
        for child in children:
            if isRoot and child["name"] in reserved_names:
                    print(f"{Fore.YELLOW}Skipping folder {reserved_names}, is VIDEO_IMAGE_PREVIEW_FOLDER{Style.RESET_ALL}")
            else:
                print(f"{Fore.LIGHTCYAN_EX}Detected folder:{Style.RESET_ALL} {child["name"]}", end="")
                if self.FolderService.exists_in_database(name=child["name"], parent=child["parent"]):
                    folder = self.FolderService.get_by_name(name=child["name"], parent=child["parent"])
                    print()
                else:
                    folder = self.FolderService.add_in_database(name=child["name"], parent=child["parent"])
                    print(Fore.GREEN, "NEW", Style.RESET_ALL)
                
                new_vids, orph = self.__discover_folder(currentFolder=child["name"], parent=folder, deleteOrphans=deleteOrphans)
                for folderId, videonames in new_vids.items():
                    new_videos[folderId] = videonames
                for folderId, orphanlist in orph.items():
                    orphans[folderId] = orphanlist
        
        return new_videos, orphans


    def clear_data(session):
        meta = db.metadata
        for table in reversed(meta.sorted_tables):
            print('Clear table %s', table)
            session.execute(table.delete())
        session.commit()
       