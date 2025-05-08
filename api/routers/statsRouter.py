import os
import time
import torch
from dotenv import load_dotenv
from flask import send_file
from flask_restful import Resource
from services.folderService import FolderService
from services.videoService import VideoService
from helpers.ValueHelper import ValueHelper

PYTORCH_MODELS_SKILLS = {
    "HAR_SA_Conv3D" : None,
    "HAR_MViT" : None,
}


load_dotenv()
STORAGE_DIR = os.getenv("STORAGE_DIR")

class StatsRouter(Resource):
    def __init__(self, **kwargs):
        self.folderService = FolderService(STORAGE_DIR)
        self.videoService = VideoService(STORAGE_DIR)
        super().__init__(**kwargs)

    def __getRecognitionResults(self, selectedModel: str):
        results = {}

        # Selected model
        results['selected-model'] = selectedModel
        results['train-time'] = 6852.3

        checkpoint = torch.load(os.path.join('..', 'computervision', 'weights', f"{selectedModel}.checkpoint.pt"), weights_only=False)
        results['f1-scores-val'] = checkpoint["f1_scores"]
        results['f1-scores-test'] = {
            "Type" : 0.51,
            "Rotations" : 0.5,
            "Turner1" : 0.5,
            "Turner2" : 0.51,
            "Skill" : 0.55,
            "Hands" : 0.5,
            "Feet" : 0.5,
            "Turntable" : 0.5,
            "BodyRotations" : 0.5,
            "Backwards" : 0.5,
            "Sloppy" : 0.5,
            "Hard2see" : 0.5,
            "Fault" : 0.5,
            "Total": 0.505,
        }
        
        f1_last_epoch = results['f1-scores-val'][len(results['f1-scores-val']) - 1]
        # All models
        results['modelcomparison'] = {
            'HAR_MViT' : {
                'accuracy': f1_last_epoch["Total"],
                'acc-skills': f1_last_epoch["Skill"],
                'last-trained' : '2025-04-28'
            },
            'HAR_MViT_extra_dense' : {
                'accuracy': 0.133,
                'acc-skills': 0.3112,
                'last-trained' : '2025-04-28'
            }
        }
        results['distributions'] = {
            'skills' : {
                'push-up': {
                    'train': 432,
                    'test': 0,
                    'val': 12
                },
                'frog': {
                    'train': 325,
                    'test': 0,
                    'val': 13
                },
                'jump': {
                    'train': 1586,
                    'test': 0,
                    'val': 132
                }
            }
        }
        

        # Best model
        results['best-model'] = selectedModel        


        return results
        {
            'best-model' : 'MViT',
            'selected-model': 'MViT',
            'train-time' : 6852.3,
            'f1-scores-val' : {
                0: {
                    "Type" : 0.5,
                    "Rotations" : 0.5,
                    "Turner1" : 0.5,
                    "Turner2" : 0.5,
                    "Skill" : 0.5,
                    "Hands" : 0.5,
                    "Feet" : 0.5,
                    "Turntable" : 0.5,
                    "BodyRotations" : 0.5,
                    "Backwards" : 0.5,
                    "Sloppy" : 0.5,
                    "Hard2see" : 0.5,
                    "Fault" : 0.5,
                    "Total": 0.5,
                },
                1: {
                    "Type" : 0.51,
                    "Rotations" : 0.5,
                    "Turner1" : 0.5,
                    "Turner2" : 0.51,
                    "Skill" : 0.55,
                    "Hands" : 0.5,
                    "Feet" : 0.5,
                    "Turntable" : 0.5,
                    "BodyRotations" : 0.5,
                    "Backwards" : 0.5,
                    "Sloppy" : 0.5,
                    "Hard2see" : 0.5,
                    "Fault" : 0.5,
                    "Total": 0.505,
                }
            },
        }
    
    def get(self):
        return {
            'localization': {
                'accuracy-mAP50' : 96.5,
                'accuracy-mAP50-95' : 66.6,
                'best-model' : 'YOLOv11',
                'train-images' : 732,
                'val-images' : 161,
                'test-images' : 130,
                'test-dd3-iou' : 0.77,
                'val-dd3-iou' : 0.81,
                'test-sr2-iou' : 0.12,
                'history' : [],
                'train-time' : 1080.1
            },
            'segmentation' : {
                'mse-val' : 0.06,
                'mse-test' : 0.065,
                'test-iou-real-over-predict' : 0.7,
                'test-iou-predict-over-real' : 0.71,
                'test-avg-splitpoint-distance' : 7.31,
                'test-avg-highest-splitpoint-distance' : 182,
                'test-highest-splitpoint-distance': 1502,
                'val-iou-real-over-predict' : 0.7,
                'val-iou-predict-over-real' : 0.71,
                'val-avg-splitpoint-distance' : 7.31,
                'val-avg-highest-splitpoint-distance' : 182,
                'val-highest-splitpoint-distance': 1502,
                'videos-train' : 44,
                'videos-val' : 7,
                'videos-test': 3,
                'total-frames': 7800,
                'best-model' : 'MViT',
                'train-time' : 6852.3
            },
            'recognition' : self.__getRecognitionResults(selectedModel='HAR_MViT')
        }, 200
