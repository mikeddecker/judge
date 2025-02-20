# Browse and navigates storage device, to find and discover videos.
# Browse and navigate storage to find DB orphans

# Nice to have : Browse and navigate storage to find renames that happend on the drive


import os
import time
import traceback
import cv2
import math
import subprocess

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
            # Make sure image folder exists
            previewfolder = os.path.join(STORAGE_DIR, VIDEO_IMAGE_PREVIEW_FOLDER)
            os.makedirs(previewfolder, exist_ok=True)

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
                    if self.VideoService.exists_in_database(name=content, folder=parent):
                        del videos_in_folder_according_to_database[content]
                        print(f"{Fore.LIGHTBLUE_EX}Detected video: {Style.RESET_ALL} {content}")
                    else:
                        print(f"{Fore.LIGHTBLUE_EX}Detected video: {Style.RESET_ALL} {content} {Fore.GREEN}NEW{Style.RESET_ALL}")
                        info = self.__enrich_video_data(name=content, folder=parent)
                        created_video_info = self.VideoService.add(name=content, folder=parent, 
                                              frameLength=info["frameLength"],
                                              width=info["width"],
                                              height=info["height"],
                                              fps=info["fps"])
                        frameNr_for_image = math.floor(info["frameLength"] * 0.2)
                        self.__create_video_image(videoId=created_video_info.Id, name=content, folder=parent, frameNr=frameNr_for_image)
                        # Bookkeeping
                        if parent.Id in new_videos.keys():
                            new_videos[parent.Id].append(content)
                        else:
                            new_videos[parent.Id] = [content]
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
        reserved_names = [VIDEO_IMAGE_PREVIEW_FOLDER, new_videos_name_key, "cropped-videos", "labeled-frames"]
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

    def __enrich_video_data(self, name: str, folder: Folder) -> dict:
        info = {
            "name" : name,
            "folderId" : folder.Id,
        }
        videopath = os.path.join(STORAGE_DIR, folder.get_relative_path(), name)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            raise IOError("Cannot open camera")
        
        info["fps"] = cap.get(cv2.CAP_PROP_FPS)
        info["frameLength"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap.release()
        cv2.destroyAllWindows()

        return info
    
    def __create_video_image(self, videoId: int, name: str, folder: Folder, frameNr: int):
        videopath = os.path.join(STORAGE_DIR, folder.get_relative_path(), name)
        cap = cv2.VideoCapture(videopath)
        if not cap.isOpened():
            raise IOError("Cannot open camera")
        
        # Create preview image
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
        res, frame = cap.read()
        filename = os.path.join(STORAGE_DIR, VIDEO_IMAGE_PREVIEW_FOLDER, f"{videoId}.jpg")
        cv2.imwrite(filename, frame)

    def __clear_data(session):
        meta = db.metadata
        for table in reversed(meta.sorted_tables):
            print('Clear table %s', table)
            session.execute(table.delete())
        session.commit()

    def download_video(self, name: str, ytid: str, folderId: int):
        ValueHelper.check_raise_string_only_abc123(name)
        ValueHelper.check_raise_id(folderId)
        folder = self.FolderService.get(folderId)
        if self.VideoService.is_already_downloaded(ytid):
            raise LookupError(f"Video already downloaded ({ytid})")
        if self.VideoService.exists_on_drive(name=name, folder=folder):
            raise LookupError(f"Videoname ({name}) already exists")
        exstension = self.__download_yt_video(
            name=name,
            ytid=ytid,
            folder=folder
        )
        print("download succesvol")
        try:
            self.__process_downloaded_video(name=f"{name}.{exstension}", folder=folder, ytid=ytid)
        except Exception as e:
            print(str(e))
            raise e
        print("processing succes")

    def __download_yt_video(self, name: str, ytid: str, folder: Folder):
        path = os.path.join(STORAGE_DIR, folder.get_relative_path(), name)
        yt_url = f"https://www.youtube.com/watch?v={ytid}"
        print("downloadinfo", name, path, yt_url)
        try:
            script_path = os.path.join(os.getcwd(), 'scripts', 'yt-download.sh')
            process = subprocess.Popen([script_path, path, yt_url],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            bufsize=0)
            while True:
                output = process.stdout.readline()
                if output.strip():
                    print(output.strip())
                if process.poll() is not None:
                    break
        except Exception as e:
            print(str(e))
            raise LookupError(f"Something went wrong with the download\n{e}")
        return 'mp4'
       
    def __process_downloaded_video(self, name: str, folder: Folder, ytid:str):
        print(name)
        info = self.__enrich_video_data(name, folder)
        created_video_info = self.VideoService.add(name=name, folder=folder, 
                                              frameLength=info["frameLength"],
                                              width=info["width"],
                                              height=info["height"],
                                              fps=info["fps"],
                                              ytid=ytid,
                                            )
        frameNr_for_image = math.floor(info["frameLength"] * 0.2)
        self.__create_video_image(videoId=created_video_info.Id, name=name, folder=folder, frameNr=frameNr_for_image)
