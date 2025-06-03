from managers.TrainerSkills import TrainerSkills
from constants import PYTORCH_MODELS_SKILLS

from helpers import load_skill_batch_X_torch, load_skill_batch_y_torch, load_segment_batch_X_torch, load_segment_batch_y_torch, adaptSkillLabels, mapBalancedSkillIndexToLabel, draw_text, calculate_splitpoint_values
from managers.DataRepository import DataRepository
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
from moviepy import ImageSequenceClip, VideoFileClip, VideoClip
import torch.nn.functional as F
from sklearn.metrics import classification_report
from pprint import pprint
import cv2

from Trainer import models, trainparams
from localizor_with_strats import predict_and_save_locations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import json
import yaml

from dotenv import load_dotenv
load_dotenv()

import sys
sys.path.append('..')
from api.helpers import ConfigHelper

import gc
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ultralytics import YOLO

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler()

STORAGE_DIR = os.getenv("STORAGE_DIR")
LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
CROPPED_VIDEOS_FOLDER = os.getenv("CROPPED_VIDEOS_FOLDER")
MODELWEIGHT_PATH = "weights"
FOLDER_VIDEORESULTS = os.getenv("FOLDER_VIDEORESULTS")


class Predictor:
    def __init__(self):
        self.repo = DataRepository()

    def predict(self, type, videoId, modelname, modelparams: dict = None, saveAsVideo:bool=False, date:str = None):
        start = time.time()
        match type:
            case 'LOCALIZE':
                self.__predict_location(videoId=videoId)
            case 'SEGMENT':
                if modelname in PYTORCH_MODELS_SKILLS.keys():
                    self.__predict_segments_pytorch(videoId=videoId,
                                                       modelname=modelname,
                                                       modelparams=modelparams)
                else:
                    raise NotImplementedError()
            case 'SKILL':
                if modelname in PYTORCH_MODELS_SKILLS.keys():
                    self.__predict_skills_pytorch(videoId=videoId,
                                                       modelname=modelname,
                                                       use_segment_predictions=False,
                                                       modelparams=modelparams,
                                                       saveAsVideo=saveAsVideo,
                                                       date=date)
                else:
                    raise NotImplementedError()
            case 'FULL':
                self.__predict_location(videoId=videoId)
                if modelname in PYTORCH_MODELS_SKILLS.keys():
                    self.__predict_skills_pytorch(videoId=videoId,
                                                       modelname=modelname,
                                                       use_segment_predictions=True,
                                                       modelparams=modelparams,
                                                       saveAsVideo=saveAsVideo,
                                                       segment_predictions=self.__predict_segments_pytorch(videoId=videoId, modelname=modelname, modelparams=modelparams),
                                                       date=date)
                else:
                    raise NotImplementedError()
            case 'SEGMENT_SKILL':
                if modelname in PYTORCH_MODELS_SKILLS.keys():
                    print("modelname", modelname)
                    self.__predict_skills_pytorch(videoId=videoId,
                                                       modelname=modelname,
                                                       use_segment_predictions=True,
                                                       modelparams=modelparams,
                                                       saveAsVideo=saveAsVideo,
                                                       segment_predictions=self.__predict_segments_pytorch(videoId=videoId, modelname="HAR_MViT_extra_dense", modelparams=modelparams),
                                                       date=date)
                else:
                    raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")
        seconds = time.time() - start
        print(f"Done, took {seconds:.1f} seconds")

    def __predict_skills_pytorch(self, videoId, modelname, use_segment_predictions, modelparams: dict = None, saveAsVideo:bool=False, segment_predictions:list = [], date:str = None):
        try:
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            
            skillconfig: dict = ConfigHelper.get_discipline_DoubleDutch_config(include_tablename=False)
            modelPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}.state_dict.pt")
            if date is not None:
                modelPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}_skills_{date}.state_dict.pt") # TODO : update in trainer
                modelPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}.state_dict.pt")

            DIM = 224
            model = PYTORCH_MODELS_SKILLS[modelname](modelinfo=modelparams, df_table_counts=self.repo.get_skill_category_counts(), skill_or_segment='skills').to(device)
            model.load_state_dict(torch.load(modelPath, weights_only=True))
            model.eval()


            balancedType = modelparams["balancedType"]
            timesteps = modelparams['timesteps']
            batch_size = modelparams['batch_size']
            assert batch_size == 1, f"Batch size must be one currently"
            frameloader = FrameLoader(self.repo)
        
            skillsInformation = None
            if use_segment_predictions:
                skillsInformation = pd.DataFrame({
                    "frameStart" : segment_predictions[:-1],
                    "frameEnd" : segment_predictions[1:]
                })
                print(skillsInformation)
            else:
                skillsInformation = self.repo.get_skills(train_test_val='val', videoId=videoId)
                skillsInformation = adaptSkillLabels(skillsInformation, balancedType)

            predictions = {}
            print(f"============= Initiation done, start predictions of video {videoId} - Using labeled segments {use_segment_predictions} =============")
            for idx in tqdm(range(len(skillsInformation))):
                skillinfo_row = skillsInformation.iloc[idx]
                frameStart = int(skillinfo_row["frameStart"])
                frameEnd = int(skillinfo_row["frameEnd"])

                batch_X, _ = load_skill_batch_X_torch(
                    frameloader=frameloader,
                    videoId=videoId,
                    dim=(DIM,DIM),
                    frameStart=frameStart,
                    frameEnd=frameEnd,
                    augment=False,
                    timesteps=timesteps,
                    normalized=False,
                )
                batch_X = batch_X.unsqueeze(dim=0)
                
                batch_y = {} if use_segment_predictions else load_skill_batch_y_torch(skillinfo_row=skillinfo_row)
                outputs = model(batch_X / 255)
                predictions[frameStart] = {}
                
                for key, pred in outputs.items():
                    target = torch.tensor([]) if use_segment_predictions else batch_y[key]

                    valueType = skillconfig[key][0]
                    if valueType == "Categorical":
                        pred = F.softmax(pred, dim=1)
                        max_idx_class, pred = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                    elif valueType == "Numerical":
                        maxValue = skillconfig[key][2]
                        pred = torch.round(pred * maxValue).type(torch.int64)
                        target = torch.round(target * maxValue).type(torch.int64)
                    else:
                        pred = torch.round(pred).type(torch.int64)
                        target = torch.round(target).type(torch.int64)
                    
                    predictions[frameStart][key] = {
                        'y_pred' : pred.item(),
                        'y_true' : None if use_segment_predictions else target.item(),
                        'y_score': None if skillconfig[key][0] != "Categorical" else max_idx_class.item(),
                        'frameEnd' : frameEnd,
                    }
            # pprint(predictions, sort_dicts=False)

            # Save predictions as JSON
            with open(os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f'{videoId}_skills_{modelname}.json'), 'w') as f:
                json.dump(predictions, f, sort_keys=True, default=str, indent=4)

            if saveAsVideo:
                videoPath = self.repo.VideoNames.loc[videoId, "name"]
                videoPath = os.path.join(STORAGE_DIR, videoPath)
                print(videoPath)
                print(f"saving predictions as a video.....")
                self.__save_skill_predictions_as_video(
                    videoId=videoId,
                    predictions=predictions,
                    balancedType=balancedType,
                    vpath=videoPath,
                    targetNames=self.repo.get_category_names(balancedType=balancedType, shiftIndex=True)
                )

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    #### Save video predictions #####################################################################################################
    def __save_skill_predictions_as_video(self, videoId:int, predictions:dict[int, dict], balancedType:str, vpath:str, targetNames:dict):
        lowerDimension = None # Manual setting for demo purposes (e.g. 720)

        cap = cv2.VideoCapture(vpath)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        videoOutputPath = os.path.join(STORAGE_DIR, FOLDER_VIDEORESULTS, f"{videoId}", f"{videoId}_annotated{"_original_size" if lowerDimension is None else ""}.mp4")
        
        # tmp_mp4 = f"{videoId}_tmp.mp4"
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(videoOutputPath, fourcc, fps, (width, height))


        if lowerDimension is not None:
            scale = lowerDimension / height
        else:
            scale = 1
        
        currentLabel = None
        def edit_frame_at_time(t, cap, currentLabel):
            skill = highfrog = hands = fault = turntable = rotations = turners = type = hard2see = sloppy = bodyRotations = ""
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_pos = 20
            fontScale = (height if lowerDimension is None else lowerDimension) / 1000
            fontThickness = height // 500
            txt_color = (0, 0, 0)
            bg_color = (0, 255, 255)
            bg_color_high = (0, 255, 255)
            bg_color_default = (137,207,240)
            bg_color_special = (255, 13, 220)

            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            if ret:
                # if pos % 500 == 0:
                #     print(f"{int(pos)}/{N}")
                
                if scale != 1:
                    frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                filteredlist = [l for start, l in predictions.items() if start <= pos and pos < l["Skill"]["frameEnd"]]
                if len(filteredlist) > 1:
                    raise ValueError(str(filteredlist))

                # print(len(filteredlist), pos, t)
                if len(filteredlist) > 0:
                    currentLabel = filteredlist[0]

                    skill = targetNames["Skill"][currentLabel["Skill"]["y_pred"]] if balancedType != 'jump_return_push_frog_other' else mapBalancedSkillIndexToLabel(balancedType=balancedType, index=currentLabel["Skill"]["y_pred"])
                    highfrog = "high" if skill == "frog" and currentLabel["Feet"]["y_pred"] == 2 else ""
                    hands = f"{currentLabel["Hands"]["y_pred"]}h"
                    turntable = f"TT{currentLabel["Turntable"]["y_pred"]}" if currentLabel["Turntable"]["y_pred"] != 0 else ""
                    rotations = f"Rotations: {currentLabel["Rotations"]["y_pred"]}"
                    turners = f"Turners: {targetNames["Turner"][currentLabel["Turner1"]["y_pred"]]} - {targetNames["Turner"][currentLabel["Turner2"]["y_pred"]]}"
                    type = f"{targetNames["Type"][currentLabel["Type"]["y_pred"]]}"
                    fault = "Fault" if currentLabel["Fault"]["y_pred"] == 1 else ""
                    hard2see = "Hard2see" if currentLabel["Hard2see"]["y_pred"] == 1 else ""
                    sloppy = "Sloppy" if currentLabel["Sloppy"]["y_pred"] == 1 else ""
                    bodyRotations = f"BodyRotations: {currentLabel["BodyRotations"]["y_pred"]}" if currentLabel["BodyRotations"]["y_pred"] != 0 else ""

                    bg_color = (0, 255, 0) if currentLabel["Skill"]["y_true"] == currentLabel["Skill"]["y_pred"] else bg_color_default
                    bg_color_high = (0, 255, 0) if currentLabel["Feet"]["y_true"] == currentLabel["Feet"]["y_pred"] else bg_color_default
                    
                    if currentLabel["Skill"]['y_true'] is None: # or currentLabel["Skill"]["y_true"] is None:
                        bg_color = (255 * (1 - currentLabel["Skill"]['y_score'] ** 2), 255 * currentLabel["Skill"]['y_score'] ** 0.5, 100 * max(0, 0.6 - currentLabel["Skill"]['y_score']))
                        bg_color_high = (0, 0, 255)

                elif currentLabel is not None and pos == currentLabel["Skill"]["frameEnd"]:
                    currentLabel = None
                    skill = highfrog = hands = fault = turntable = rotations = turners = type = hard2see = sloppy = bodyRotations = ""

                
                horizontal = 0
                vertical = 0
                w, h = draw_text(frame, type, pos=(text_pos, text_pos), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                
                w, h = draw_text(frame, hands, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                horizontal += w
                w, h = draw_text(frame, highfrog, pos=(text_pos + horizontal, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                horizontal += w
                w, h = draw_text(frame, turntable, pos=(text_pos + horizontal, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                horizontal += w
                w, h = draw_text(frame, skill, pos=(text_pos + horizontal, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                
                w, h = draw_text(frame, turners, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                w, h = draw_text(frame, rotations, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                w, h = draw_text(frame, bodyRotations, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                w, h = draw_text(frame, hard2see, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                w, h = draw_text(frame, sloppy, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)
                w, h = draw_text(frame, fault, pos=(text_pos, text_pos + vertical), font=font, font_scale=fontScale, font_thickness=fontThickness, text_color=txt_color, text_color_bg=bg_color_default)
                vertical += int(1.5*h)

                # frames.append(frame)
                # out.write(frame)
                return frame
            else:
                return np.zeros(shape=(width,height,3))
        
        # out.release()
        
        clip = VideoClip(lambda t: edit_frame_at_time(t, cap, currentLabel), duration=N/fps)
        clip: VideoClip = clip.with_audio(VideoFileClip(vpath).audio)
        clip.write_videofile(videoOutputPath, codec='libx264', fps=fps)
        cap.release()
        # clip = ImageSequenceClip(frames, (tmp_mp4)
        # clip.write_videofile(videoOutputPath, codec='libx264')
        # os.remove(tmp_mp4)

    def __predict_segments_pytorch(self, videoId, modelname, modelparams: dict = None):
        try:
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            # TODO : update to use best val checkpoint 
            modelPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}_segmentation.state_dict.pt")

            DIM = 224
            model = PYTORCH_MODELS_SKILLS[modelname](skill_or_segment='segments', modelinfo=modelparams, df_table_counts=self.repo.get_skill_category_counts()).to(device)
            model.load_state_dict(torch.load(modelPath, weights_only=True))
            model.eval()

            timesteps = modelparams['timesteps']
            batch_size = modelparams['batch_size']
            assert batch_size == 1, f"Batch size must be one currently"
            frameloader = FrameLoader(self.repo)
        
            videoInfo = self.repo.get_videoinfo(videoId)
            frameLength = videoInfo.loc[0, "frameLength"]
            fps = videoInfo.loc[0, "fps"]
            timesteps = modelparams["timesteps"]
            offset = (frameLength % timesteps) // 2
            batches = frameLength // timesteps
            labeledSkills = self.repo.get_skills(train_test_val='val', videoId=videoId)

            split_threshold = 0.4
            df_splitpoint_values = calculate_splitpoint_values(
                videoId=videoId,
                frameLength=frameLength,
                df_Skills=labeledSkills,
                fps=fps,
                Nsec_frames_around=1/6
            )

            targets = [0 for _ in range(frameLength)]
            predictions = [0 for _ in range(frameLength)]
            print(f"============= Initiation done, start segment predictions for video {videoId} =============")
            for idx in tqdm(range(batches)):
                frameStart = idx * timesteps + offset
                frameEnd = frameStart + timesteps

                batch_X = load_segment_batch_X_torch(
                    frameloader=frameloader,
                    videoId=videoId,
                    dim=(DIM,DIM),
                    frameStart=frameStart,
                    frameEnd=frameEnd,
                    augment=False,
                    timesteps=timesteps,
                    normalized=False,
                ).unsqueeze(dim=0) # Unsqueeze(dim=0) = add batch dimension

                batch_y = load_segment_batch_y_torch(
                    frameStart=frameStart,
                    frameEnd=frameEnd,
                    df_splitpoint_values=df_splitpoint_values
                )

                outputs = model(batch_X / 255)[0] # [0] Remove batch size

                targets[frameStart:frameEnd] = batch_y.tolist()
                predictions[frameStart:frameEnd] = outputs.tolist()
                
            points = 250
            for startIdx in range(0, frameLength, points):
                if startIdx + points > frameLength:
                    points = frameLength % points
                fig, ax1 = plt.subplots()

                # Plot the first y-axis data
                color = 'tab:red'
                ax1.set_ylabel('pred', color=color)
                ax1.plot(range(startIdx, points+startIdx), predictions[startIdx:startIdx+points], color=color)
                ax1.tick_params(axis='y', labelcolor=color)

                # Create a second y-axis that shares the same x-axis
                ax2 = ax1.twinx()

                # Plot the second y-axis data
                color = 'tab:blue'
                ax2.set_ylabel('target', color=color)
                ax2.plot(range(startIdx, points+startIdx), targets[startIdx:startIdx+points], color=color)
                ax2.tick_params(axis='y', labelcolor=color)
                os.makedirs(os.path.join("plots"), exist_ok=True)
                plotpath = os.path.join("plots", f"segmentplot_{videoId}_frameStart_{startIdx}_with_{points}_points.png")
                plt.savefig(plotpath)
            plt.close()

            predictions = np.array(predictions)
            predictions_bigger_than_split_threshold = np.where(predictions > split_threshold, predictions, 0)
            p_split = predictions_bigger_than_split_threshold
            window_size = int(fps // 3)
            predictions_argMax_in_window = [s - window_size + np.argmax(p_split[max(0, s-window_size):min(frameLength, s+window_size)]) for s in range(frameLength)]
            predictions_splitmoments = np.where(predictions > split_threshold, predictions_argMax_in_window, 0)
            predictions_splitmoments = np.unique(predictions_splitmoments)
            
            distances = predictions_splitmoments[1:] - predictions_splitmoments[:-1]
            predictions_splitmoments = predictions_splitmoments[1:]
            predictions_splitmoments = predictions_splitmoments[np.where(distances < window_size // 3, False, True)]
            predictions_splitmoments = [int(g) for g in predictions_splitmoments]

            print([f"{d:4d}" for d in predictions_splitmoments])
            print(f"Predicted {len(predictions_splitmoments) - 1} splitpoints (end not included)")

            return predictions_splitmoments

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()
      
    
    def __predict_location(self, videoId):
        modelname, modelpath = ConfigHelper.localize_get_best_modelpath()

        predict_and_save_locations(
            modeldir=modelpath,
            repo=self.repo,
            modelname=modelname,
            videoIds=[videoId]
        )

##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

modelparams = {
    "HAR_MViT" : {
        "balancedType" : "limit_10procent", # jump_return_push_frog_other
        "dim" : 224,
        "timesteps" : 16,
        "batch_size" : 1,
    }
}

if __name__ == "__main__":
    predictor = Predictor()

    videoIds = [1285, 1315, 1408, 2283, 2285, 2289, 2288, 2296, 2309, 2568,2569,2570,2571,2572,2573,2574,2575,2576,2577,2578,2579,2580,2581,2582,2583,2584,2585,2586,2587,2588,2589,]
    videoIds = range(2568, 2590)
    videoIds = [2749, 2776]
    models = ['HAR_SwinT_s']
    dates = ["20250525", "20250524"]
    dates = ["20250525"]

    for d in dates:
        for modelname in models:
            for videoId in videoIds:
                # predictor.predict(
                #     type="LOCALIZE",
                #     videoId=videoId,
                #     modelname=None,
                # )
                predictor.predict(
                    type="SEGMENT_SKILL",
                    videoId=videoId,
                    modelname=modelname,
                    modelparams=trainparams[modelname],
                    saveAsVideo=True,
                    date=d
                )
                # TODO : cache segment predictions or save them as json

    # predictor.predict(
    #     type="SKILL",
    #     videoId=1315,
    #     modelname=modelname,
    #     modelparams=modelparams,
    #     saveAsVideo=True,
    # )

    # predictor.predict(
    #     type="SKILL",
    #     videoId=2285,
    #     modelname=modelname,
    #     modelparams=modelparams,
    #     saveAsVideo=True,
    # )

    # predictor.predict(
    #     type="SEGMENT",
    #     videoId=2305,
    #     modelname=modelname,
    #     modelparams=modelparams,
    #     saveAsVideo=True,
    # )
