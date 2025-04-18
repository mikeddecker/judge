from managers.TrainerSkills import TrainerSkills
from constants import PYTORCH_MODELS_SKILLS

from helpers import load_skill_batch_X_torch, load_skill_batch_y_torch, load_segment_batch_X_torch, load_segment_batch_y_torch, adaptSkillLabels, mapBalancedSkillIndexToLabel, draw_text, calculate_splitpoint_values
from managers.DataRepository import DataRepository
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
from moviepy import ImageSequenceClip
import torch.nn.functional as F
from sklearn.metrics import classification_report
from pprint import pprint
import cv2

from dotenv import load_dotenv
load_dotenv()

import gc
import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler()

STORAGE_DIR = os.getenv("STORAGE_DIR")
LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
CROPPED_VIDEOS_FOLDER = os.getenv("CROPPED_VIDEOS_FOLDER")
MODELWEIGHT_PATH = "weights"


class Predictor:
    def predict(self, type, videoId, modelname, modelparams: dict = None, saveAsVideo:bool=False):
        match type:
            case 'LOCALIZE':
                raise NotImplementedError()
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
                                                       saveAsVideo=saveAsVideo)
                else:
                    raise NotImplementedError()
            case _:
                raise ValueError(f"Trainer - Type {type} not recognized")

    def __predict_skills_pytorch(self, videoId, modelname, use_segment_predictions, modelparams: dict = None, saveAsVideo:bool=False):
        try:
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            if use_segment_predictions:
                raise NotImplementedError()
            
            # TODO : update to use best val checkpoint 
            modelPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}.state_dict.pt")

            DIM = 224
            repo = DataRepository()
            model = PYTORCH_MODELS_SKILLS[modelname](modelinfo=modelparams, df_table_counts=repo.get_skill_category_counts(), skill_or_segment='skills').to(device)
            model.load_state_dict(torch.load(modelPath, weights_only=True))
            model.eval()


            timesteps = modelparams['timesteps']
            batch_size = modelparams['batch_size']
            assert batch_size == 1, f"Batch size must be one currently"
            frameloader = FrameLoader(repo)
        
            balancedType = modelparams["balancedType"]
            labeledSkills = repo.get_skills(train_test_val='val', videoId=videoId)
            labeledSkills = adaptSkillLabels(labeledSkills, balancedType)

            predictions = {}
            print(f"============= Initiation done, start predictions of video {videoId} =============")
            for idx in tqdm(range(len(labeledSkills))):
                skillinfo_row = labeledSkills.iloc[idx]
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
                ).unsqueeze(dim=0)
                batch_y = load_skill_batch_y_torch(skillinfo_row=skillinfo_row)
                outputs = model(batch_X / 255)
                pred = F.softmax(outputs["Skill"], dim=1)
                max_score, max_idx_class = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                predictions[frameStart] = {
                    'y_pred' : max_idx_class.item(),
                    'y_true' : batch_y["Skill"].item(),
                    'y_score': max_score.item(),
                    'isCorrect' : (max_idx_class == batch_y["Skill"]).item(),
                    'frameEnd' : frameEnd,
                }
            print()
            pprint(predictions, sort_dicts=False)

            if saveAsVideo:
                videoPath = repo.VideoNames.loc[videoId, "name"]
                videoPath = os.path.join(STORAGE_DIR, videoPath)
                print(videoPath)
                print(f"saving predictions as a video.....")
                self.__save_skill_predictions_as_video(
                    videoId=videoId,
                    predictions=predictions,
                    balancedType=balancedType,
                    vpath=videoPath,
                    targetNames=repo.get_category_names(balancedType=balancedType)
                )

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()

#### Save video predictions #####################################################################################################
    def __save_skill_predictions_as_video(self, videoId:int, predictions:dict, balancedType:str, vpath:str, targetNames:dict):
        cap = cv2.VideoCapture(vpath)
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        scale = 0.4
        ret, frame = cap.read()
        frames = []
        skill = ""
        endFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        currentLabel = None
        videoOutputPath = os.path.join(STORAGE_DIR, "annotated-videos", f"{videoId}.mp4")
        os.makedirs(os.path.join(STORAGE_DIR, "annotated-videos"), exist_ok=True)
        
        # tmp_mp4 = f"{videoId}_tmp.mp4"
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter(videoOutputPath, fourcc, fps, (width, height))


        font = cv2.FONT_HERSHEY_SIMPLEX
        text_pos = (50, 50)
        fontScale = 2
        txt_color = (0, 0, 0)
        bg_color = (0, 255, 255)
        thickness = 2
        while ret:
            if pos % 500 == 0:
                print(f"{int(pos)}/{N}")
            frame = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # draw_text(frame, )
            if pos in predictions.keys():
                currentLabel = predictions[pos]
                skill = targetNames["Skill"][currentLabel["y_pred"]] if balancedType != 'jump_return_push_frog_other' else mapBalancedSkillIndexToLabel(balancedType=balancedType, index=currentLabel["y_pred"])
                bg_color = (0, 255, 0) if currentLabel["isCorrect"] else (255, 20, 0)
            elif currentLabel is not None and pos == currentLabel["frameEnd"]:
                currentLabel = None
                skill = ""

            w, h = draw_text(frame, skill, pos=text_pos, font=font, font_scale=fontScale, text_color=txt_color, text_color_bg=bg_color)
            frames.append(frame)
            # out.write(frame)
            
            pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = cap.read()
        # out.release()
        
        clip = ImageSequenceClip(frames, fps=fps)
        clip.write_videofile(videoOutputPath, codec='libx264')
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
            repo = DataRepository()
            model = PYTORCH_MODELS_SKILLS[modelname](skill_or_segment='segments', modelinfo=modelparams, df_table_counts=repo.get_skill_category_counts()).to(device)
            model.load_state_dict(torch.load(modelPath, weights_only=True))
            model.eval()

            timesteps = modelparams['timesteps']
            batch_size = modelparams['batch_size']
            assert batch_size == 1, f"Batch size must be one currently"
            frameloader = FrameLoader(repo)
        
            videoInfo = repo.get_videoinfo(videoId)
            frameLength = videoInfo.loc[0, "frameLength"]
            fps = videoInfo.loc[0, "fps"]
            timesteps = modelparams["timesteps"]
            offset = (frameLength % timesteps) // 2
            batches = frameLength // timesteps
            labeledSkills = repo.get_skills(train_test_val='val', videoId=videoId)
            # labeledSkills = adaptSkillLabels(labeledSkills, balancedType)

            window_size = 7
            split_threshold = 0.5
            df_splitpoint_values = calculate_splitpoint_values(
                videoId=videoId,
                frameLength=frameLength,
                df_Skills=labeledSkills,
                fps=fps,
                Nsec_frames_around=1/window_size
            )

            predictions = {}
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
                
                print('targets', batch_y)
                print('-'*50)
                print('outputs', outputs)
                
            print()
            pprint(predictions, sort_dicts=False)

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()


##################################################################################################################################
##################################################################################################################################
##################################################################################################################################

if __name__ == "__main__":
    modelparams = {
        # "balancedType" : "jump_return_push_frog_other",
        "balancedType" : "limit_10procent",
        "dim" : 224,
        "timesteps" : 16,
        "batch_size" : 1,
    }
    modelname = "HAR_SA_Conv3D"
    modelname = "HAR_MViT"
    predictor = Predictor()

    predictor.predict(
        type="SKILL",
        videoId=1315,
        modelname=modelname,
        modelparams=modelparams,
        saveAsVideo=True,
    )

    predictor.predict(
        type="SKILL",
        videoId=2285,
        modelname=modelname,
        modelparams=modelparams,
        saveAsVideo=True,
    )

    predictor.predict(
        type="SEGMENT",
        videoId=1315,
        modelname=modelname,
        modelparams=modelparams,
        saveAsVideo=True,
    )
