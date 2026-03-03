# Browse and navigates storage device, to find and discover videos.
# Browse and navigate storage to find DB orphans

# Nice to have : Browse and navigate storage to find renames that happend on the drive

import os
import time
import traceback
import cv2
import math
import random
import subprocess

from pprint import pprint

from config import ENVS
from colorama import Fore, Style
from domain.folder import Folder
from domain.tag import Tag
from domain.videoinfo import VideoInfo
from helpers.ValueHelper import ValueHelper
from services.videoService import VideoService
from services.folderService import FolderService
from services.tagService import TagService
from repository.db import db

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
    def __init__(self):
        ValueHelper.check_raise_string(ENVS.DIRS.VIDEOS)
        if not os.path.exists(ENVS.DIRS.VIDEOS):
            raise NotADirectoryError(f"StorageFolder {ENVS.DIRS.VIDEOS} does not exist")
        self.VideoService = VideoService()
        self.FolderService = FolderService()
        self.TagService = TagService()

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
            print(f"{Fore.YELLOW}Discovering folder:{Style.RESET_ALL}", f"{ENVS.DIRS.VIDEOS} (root)")
            new_videos, orphans = self.__discover_folder(ENVS.DIRS.VIDEOS, parent=None, isRoot=True, deleteOrphans=deleteOrphans)
            print(f"{Fore.GREEN}Done {Style.RESET_ALL} exploring drive{Style.RESET_ALL}")

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

    def __discover_folder(self, currentFolder: str, parent: Folder, isRoot=False, deleteOrphans=False, folderTags: set[Tag]= set()):
        if currentFolder is None or not isinstance(currentFolder, str):
            raise ValueError(f"Didn't get a string for folder, got", currentFolder)
        if not isRoot and (parent is None or not isinstance(parent, Folder)):
            raise ValueError(f"Not in root folder, but got no parent, got {parent} instead")
        
        currentFolderPath = currentFolder if isRoot else os.path.join(ENVS.DIRS.VIDEOS, parent.get_relative_path())
        folder_content = os.listdir(currentFolderPath)
        videos_in_folder_according_to_database = {} if isRoot else self.VideoService.get_videos(parent.Id)
        videos_in_folder_according_to_database = { videoinfo.Name : videoinfo for videoinfo in videos_in_folder_according_to_database }
        child_folders : list[dict] = []
        new_videos = {}
        orphans = {}

        tags = self.TagService.get_tags()

        for content in folder_content:
            contentPath = os.path.join(currentFolderPath, content)

            # Rename files with spaces
            if content.__contains__(" ") or content.__contains__("_"):
                old_name = contentPath
                contentPath = contentPath.replace(" ", "-").replace("_", "-")
                content = content.replace(" ", "-").replace("_", "-")
                os.rename(old_name, contentPath)
                print(f"{Fore.MAGENTA}File or folder contains spaces and underscores, renamed with (-) dashes:{Style.RESET_ALL}", content)

            # Temp save dirs, to provide better output
            # Otherwise videos, and folders interlap with each other
            if os.path.isdir(contentPath):
                child_folders.append({"name": content, "parent": parent})

            # Handle files
            if os.path.isfile(contentPath):
                if isRoot:
                    print(f"{Fore.YELLOW}Skipping file in root:{Style.RESET_ALL} {content}")
                elif content.split(".")[-1] in ENVS.SUPPORTED_VIDEO_FORMATS:
                    if self.VideoService.exists_in_database(name=content, folder=parent):
                        del videos_in_folder_according_to_database[content]
                        videoId : int = self.VideoService.get_videoId(name=content, folder=parent)
                        detected_tags : set[Tag] = self.__filename_to_tags(filename=content, tags=tags)
                        for tag in detected_tags:
                            self.VideoService.add_tag(videoId=videoId, tag=tag)
                        for tag in folderTags:
                            self.VideoService.add_tag(videoId=videoId, tag=tag)

                        # Create video image if not exists
                        videoinfo : VideoInfo = self.VideoService.get(videoId)
                        frameNr_for_image = math.floor(videoinfo.FrameLength * (0.2 + random.random() / 2))
                        self.__create_video_image(videoId=videoId, name=content, folder=parent, frameNr=frameNr_for_image)

                    else:
                        print(f"{Fore.LIGHTBLUE_EX}Detected video: {Style.RESET_ALL} {content} {Fore.GREEN}NEW{Style.RESET_ALL}")
                        info = self.__enrich_video_data(name=content, folder=parent, tags=tags)

                        inserted_video = self.VideoService.add(name=content, folder=parent,
                            frameLength=info["frameLength"],
                            width=info["width"],
                            height=info["height"],
                            fps=info["fps"],
                            tags=info["tags"]
                        )

                        # Bookkeeping
                        if parent.Id in new_videos.keys():
                            new_videos[parent.Id].append(content)
                        else:
                            new_videos[parent.Id] = [content]

                        # Create video image
                        frameNr_for_image = math.floor(info["frameLength"] * (0.2 + random.random() / 2))
                        self.__create_video_image(videoId=inserted_video.Id, name=content, folder=parent, frameNr=frameNr_for_image)
                elif content.split(".")[-1] in ENVS.SUPPORTED_IMAGE_FORMATS:
                    print(f"{Fore.LIGHTMAGENTA_EX}Detected image:{Style.RESET_ALL} {content} (currently skipped)")
                else:
                    print(f"{Fore.YELLOW}Detected other:{Style.RESET_ALL} {content}")

        for orpan_name, videoinfo in videos_in_folder_according_to_database.items():
            print(f"{Fore.RED}Detected orphan: videoId = {Fore.YELLOW}{videoinfo.Id}{Style.RESET_ALL} {orpan_name}")
            if deleteOrphans:
                self.VideoService.delete_from_database(id=videoinfo.Id)
            if parent.Id in orphans.keys():
                orphans[str(parent.Id)][str(videoinfo.Id)] = orpan_name
            else:
                orphans[str(parent.Id)] = { videoinfo.Id : orpan_name }

        # Now loop al folders in current folder
        for child in child_folders:
            print(f"{Fore.LIGHTCYAN_EX}Exploring folder:{Style.RESET_ALL} {child["name"]}", end="")
            if self.FolderService.exists_in_database(name=child["name"], parent=child["parent"]):
                folder = self.FolderService.get_by_name(name=child["name"], parent=child["parent"])
                print()
            else:
                folder = self.FolderService.add_in_database(name=child["name"], parent=child["parent"])
                print(Fore.GREEN, "NEW", Style.RESET_ALL)

            subFolderTags = folderTags.union(self.__filename_to_tags(filename=child["name"], tags=tags))

            new_vids, orph = self.__discover_folder(currentFolder=child["name"], parent=folder, deleteOrphans=deleteOrphans, folderTags=subFolderTags)
            for folderId, videonames in new_vids.items():
                new_videos[folderId] = videonames
            for folderId, orphanlist in orph.items():
                orphans[folderId] = orphanlist

        return new_videos, orphans

    def __enrich_video_data(self, name: str, folder: Folder, tags: list[Tag], tagOnly:bool=False) -> dict:
        info = {
            "name" : name,
            "folderId" : folder.Id,
            "tags": self.__filename_to_tags(filename=name, tags=tags)
        }
        if not tagOnly:
            videopath = os.path.join(ENVS.DIRS.VIDEOS, folder.get_relative_path(), name)
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

    def __filename_to_tags(self, filename: str, tags: list[Tag]) -> set[Tag]:
        detected_tags: set[Tag] = set()
        parts = filename.lower().split('.')[0].split("-")
        for tag in tags:
            for video_name_part in parts:
                if tag.contains_keyword(video_name_part):
                    detected_tags.add(tag)
        return detected_tags

    def __create_video_image(self, videoId: int, name: str, folder: Folder, frameNr: int):
        # Make sure videofolder exists, for storing predictions, image...
        inserted_videofolder = os.path.join(ENVS.DIRS.GENERATED_VIDEODATA, f"{videoId}")
        os.makedirs(inserted_videofolder, exist_ok=True)

        image_filename = os.path.join(inserted_videofolder, f"{videoId}.jpg")
        # Load video
        videopath = os.path.join(ENVS.DIRS.VIDEOS, folder.get_relative_path(), name)
        if not os.path.exists(image_filename):
            cap = cv2.VideoCapture(videopath)
            if not cap.isOpened():
                raise IOError("Cannot open camera")

            # Create preview image
            cap.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
            _, frame = cap.read()
            cv2.imwrite(image_filename, frame)
            print(f"{Fore.LIGHTMAGENTA_EX}Created image:{Style.RESET_ALL} {name}")

    def __clear_data(session):
        meta = db.metadata
        for table in reversed(meta.sorted_tables):
            print('Clear table %s', table)
            session.execute(table.delete())
        session.commit()

    def download_video(self, name: str, ytid: str, folderId: int):
        ValueHelper.check_raise_string_only_abc123(name)
        ValueHelper.check_raise_uuid(folderId)
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
        path = os.path.join(ENVS.DIRS.VIDEOS, folder.get_relative_path(), name)
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
        print(f"Processing downloaded:", name)
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

