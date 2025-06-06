import os
from dotenv import load_dotenv
from flask import send_file, Response, request
from flask_restful import Resource, current_app
from domain.frameinfo import FrameInfo
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class FrameRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def post(self, videoId: int, frameNr: int):
        data = request.get_json()
        
        # Extract the required fields from the body
        # frameNr = data.get('frameNr')
        x = data.get('x')
        y = data.get('y')
        width = data.get('width')
        height = data.get('height')
        jumperVisible = data.get('jumperVisible', True)  # Default to True if not provided
        labeltype = data.get('labeltype')
        try:
            ValueHelper.check_raise_id(videoId)
            ValueHelper.check_raise_frameNr(frameNr)
            ValueHelper.check_float_between_0_and_1_inclusive(x)
            ValueHelper.check_float_between_0_and_1_inclusive(y)
            ValueHelper.check_float_between_0_and_1_inclusive(width)
            ValueHelper.check_float_between_0_and_1_inclusive(height)
        except ValueError as ve:
            return str(ve), 404
        frameinfo = FrameInfo(frameNr=frameNr, x=x, y=y, width=width, height=height, jumperVisible=jumperVisible, labeltype=labeltype)
        video = self.videoService.get(videoId)
        video = self.videoService.set_frameInfo(frameInfo=frameinfo, video=video)
        return video.to_dict(), 200
    
    def delete(self, videoId: int, frameNr: int):
        data = request.get_json()
        frameinfo = data.get('frameinfo')
        try:
            ValueHelper.check_raise_id(videoId)
            ValueHelper.check_raise_frameNr(frameNr)
        except ValueError as ve:
            return str(ve), 404
        videoinfo = self.videoService.get(videoId)
        frameinfo = FrameInfo(frameNr=frameNr, x=frameinfo["X"], y=frameinfo["Y"],
                              width=frameinfo["Width"], height=frameinfo["Height"],
                              jumperVisible=frameinfo["JumperVisible"], labeltype=frameinfo["LabelType"])
        return self.videoService.remove_frameInfo(frameNr=frameNr, video=videoinfo, frameinfo=frameinfo).to_dict(), 200

