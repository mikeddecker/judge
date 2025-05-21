import os
from dotenv import load_dotenv
from flask import send_file, Response, jsonify, request
from flask_restful import Resource, current_app
from services.folderService import FolderService
from services.videoService import VideoService
from services.jobService import JobService
from helpers.ValueHelper import ValueHelper
import cv2

load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")
FOLDER_VIDEORESULTS = os.getenv("FOLDER_VIDEORESULTS")
CROPPED_VIDEOS_FOLDER = "cropped-videos"
CROPPED_VIDEOS_STATUSES = ["OK", "OK_NET_NIET_PERFECT", "SLECHT"]



class JobTrainRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        self.jobService = JobService(STORAGE_DIR)
        super().__init__(**kwargs)
    
    def get(self):
        self.jobService.re_train_and_predict()
        return "Ok", 200
