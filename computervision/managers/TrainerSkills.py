import gc
import os
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim

from constants import PYTORCH_MODELS_SKILLS
from dotenv import load_dotenv
from managers.DataRepository import DataRepository
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import sys
sys.path.append('..')
from api.helpers import ConfigHelper

load_dotenv()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler()

STORAGE_DIR = os.getenv("STORAGE_DIR")
LABELS_FOLDER = "labels"
SUPPORTED_VIDEO_FORMATS = os.getenv("SUPPORTED_VIDEO_FORMATS")
CROPPED_VIDEOS_FOLDER = os.getenv("CROPPED_VIDEOS_FOLDER")
MODELWEIGHT_PATH = "weights"

class TrainerSkills:
    def __compute_losses(self, outputs, batch_y, loss_fns):
        """Outputs is a ModelDict, acts like a dictionary"""
        losses = []

        for key, pred in outputs.items():
            target = batch_y[key]
            
            if key in ['Skill', 'Turner1', 'Turner2', 'Type']:  # Categorical
                loss = loss_fns['categorical'](pred, target.long())
            else:
                loss = loss_fns['regression'](pred.squeeze(), target)
            
            losses.append(loss)
        return sum(losses)

    def validate(self, model, dataloader, optimizer, loss_fns, target_names, device='cuda'):
        model.eval()
        val_loss = 0.0

        skillconfig: dict = ConfigHelper.get_discipline_DoubleDutch_config(include_tablename=False)
        y_pred = { key : [] for key, _ in skillconfig.items() }
        y_true = { key : [] for key, _ in skillconfig.items() }

        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader):
                with torch.amp.autocast(device_type='cuda'):
                    optimizer.zero_grad()
                    outputs = model(batch_X / 255)
                    # Loss
                    total_batch_loss = self.__compute_losses(outputs=outputs, batch_y=batch_y, loss_fns=loss_fns)
                    val_loss += total_batch_loss.item()

                    # Accuracy
                    for key, pred in outputs.items():
                        target = batch_y[key]

                        valueType = skillconfig[key][0]
                        if valueType == "Categorical":
                            pred = F.softmax(pred, dim=1)
                            _, pred = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                        elif valueType == "Numerical":
                            maxValue = skillconfig[key][2]
                            pred = torch.round(pred * maxValue).squeeze(dim=0).type(torch.int64)
                            target = torch.round(target * maxValue).type(torch.int64)
                        else:
                            pred = torch.round(pred).squeeze(dim=0).type(torch.int64)
                            target = torch.round(target).type(torch.int64)
                        
                        y_pred[key].extend(pred.data.cpu().numpy())
                        y_true[key].extend(target.data.cpu().numpy())
                    
        
        print(f"="*80)
        classification_reports = {}
        for key in y_true.keys():
            classKey = key if key not in ['Turner1', 'Turner2'] else 'Turner'
            labels = None if classKey not in target_names.keys() else range(1, len(target_names[classKey]) + 1) # Mysql startIdx = 1
            tn = None if classKey not in target_names.keys() else target_names[classKey]
            classification_reports_string = classification_report(y_true[key], y_pred[key], labels=labels, target_names=tn, zero_division=0)
            classification_reports[key] = classification_report(y_true[key], y_pred[key], output_dict=True, labels=labels, target_names=tn, zero_division=0)
            print(f"----- Details {key} ----")
            print(classification_reports_string)

            lbls = labels if labels is not None else range(max(max(y_true[key]), max(y_pred[key])) + 1)
            cm = confusion_matrix(y_true[key], y_pred[key], labels=lbls)
            cm_df = pd.DataFrame(cm, index=labels, columns=labels)

            print("Confusion Matrix:", key)
            print(cm_df)
            print(f"="*80)

        f1_scores_epoch = { k: class_report['macro avg']['f1-score'] for k, class_report in classification_reports.items() }
        f1_scores_epoch["Total"] = sum(f1_scores_epoch.values()) / len(f1_scores_epoch)

        print(f"Total skill (macro avg) accuracy", classification_reports['Skill']['macro avg'])

        return val_loss / len(dataloader), f1_scores_epoch, classification_reports, cm

    def train(self, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, trainparams: dict= {}, learning_rate=1e-5):
        try:
            testrun = False
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            path = os.path.join(MODELWEIGHT_PATH, f"{modelname}.state_dict.pt")
            checkpointPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}{'_testrun' if testrun else ''}.checkpoint.pt")

            
            DIM = 224
            repo = DataRepository()
            model = PYTORCH_MODELS_SKILLS[modelname](skill_or_segment="skills", modelinfo=trainparams, df_table_counts=repo.get_skill_category_counts()).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)
            epoch_start = 0
            f1_scores = {}
            classification_reports = {}
            losses = []
            if not from_scratch and os.path.exists(checkpointPath):
                checkpoint = torch.load(checkpointPath, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_start = checkpoint['epoch'] + 1
                losses = checkpoint['losses']
                f1_scores = {} if 'f1_scores' not in checkpoint.keys() else checkpoint['f1_scores']
                classification_reports = {} if 'classification_reports' not in checkpoint.keys() else checkpoint['classification_reports']

            if unfreeze_all_layers:
                for param in model.parameters():
                    param.requires_grad = True

            train_generator = DataGeneratorSkills(
                frameloader=FrameLoader(repo),
                train_test_val="train",
                dim=(DIM,DIM),
                timesteps=trainparams['timesteps'],
                batch_size=trainparams['batch_size'],
                testrun=testrun
            )
            val_generator = DataGeneratorSkills(
                frameloader=FrameLoader(repo),
                train_test_val="val",
                dim=(DIM,DIM),
                timesteps=trainparams['timesteps'],
                batch_size=trainparams['batch_size'],
                testrun=testrun
            )
        
            dataloaderTrain = DataLoader(train_generator, batch_size=1, shuffle=True)
            dataloaderVal = DataLoader(val_generator, batch_size=1, shuffle=True)

            loss_fns = {
                'categorical': torch.nn.CrossEntropyLoss(),
                'regression': torch.nn.MSELoss()
            }
            balancedType = trainparams["balancedType"]
            target_names = repo.get_category_names(balancedType=balancedType)

            # Training loop
            for epoch in range(epoch_start, epochs + epoch_start):
                print(f"============= EPOCH {epoch} =============")
                model.train()
                total_loss = 0.0
                i = 0
                for batch_X, batch_y in tqdm(dataloaderTrain):
                    with torch.amp.autocast(device_type='cuda'):
                        optimizer.zero_grad()  # Clear gradients
                        
                        # Forward pass
                        outputs = model(batch_X / 255)
                        total_batch_loss = self.__compute_losses(outputs=outputs, batch_y=batch_y, loss_fns=loss_fns)
                        total_batch_loss.backward()
                        optimizer.step()
                    
                    total_loss += total_batch_loss.item()
                    i+=1

                print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloaderTrain):.4f}")

                val_loss, f1_scores_epoch, class_reports, conf_matrix = self.validate(model=model, dataloader=dataloaderVal, optimizer=optimizer, loss_fns=loss_fns, target_names=target_names)
                losses.append(val_loss)
                scheduler.step(val_loss)
                f1_scores[epoch] = f1_scores_epoch
                classification_reports[epoch] = class_reports
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f} (val loss = {val_loss})")
                
                minIndex = losses.index(min(losses))
                epochsNoImprovement = len(losses) - minIndex - 1
                hasValLossImproved = epochsNoImprovement == 0

                if epochsNoImprovement > 2:
                    print(f"No improvement for {epochsNoImprovement} - stopping")
                    break

                if hasValLossImproved:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'losses': losses,
                        'f1_scores': f1_scores,
                        'classification_reports' : classification_reports,
                        'final_classification_reports' : class_reports,
                        'confusion_matrix': conf_matrix,
                    }, checkpointPath)
            
                    torch.save(model.state_dict(), path)
            pprint(f1_scores)

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()
