from constants import PYTORCH_MODELS_SKILLS
from managers.DataRepository import DataRepository
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
import torch.nn.functional as F
from sklearn.metrics import classification_report

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

class TrainerSkills:
    def __compute_losses(self, outputs, batch_y, loss_fns):
        # Compute losses for each output head
        losses = []

        for key, pred in outputs.items():
            target = batch_y[key]
            
            if key in ['Skill', 'Turner1', 'Turner2', 'Type']:  # Categorical
                loss = loss_fns['categorical'](pred, target.long())
                if key == 'Skill':
                    loss *= 10
            else:  # Regression
                loss = loss_fns['regression'](pred.squeeze(), target)
            
            losses.append(loss)
        
        # Total loss (sum of all individual losses)
        return sum(losses)

    def __compute_accuracy(self, outputs, targets):
        # Compute losses for each output head
        correctCounts = {
            'Skill' : 0, 
            'Turner1': 0,
            'Turner2': 0, 
            'Type' : 0,
        }
        for key, pred in outputs.items():
            target = targets[key]

            
            if key in ['Skill', 'Turner1', 'Turner2', 'Type']:  # Categorical
                pred = F.softmax(pred, dim=1)
                max_scores, max_idx_class = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                # print(max_scores)
                # print(max_idx_class)
                print(key, max_idx_class, max_scores)
                print(target)
                acc = (max_idx_class == target).sum().item() / max_scores.size(0)
                # print(f"Acc for {key} is {acc}")
                
                correctCounts[key] += acc
            # else:  # Regression
            #     loss = loss_fns['regression'](pred.squeeze(), target)
            
        
        # Total loss (sum of all individual losses)
        return correctCounts

    def validate(self, model, dataloader, optimizer, loss_fns, device='cuda'):
        model.eval()
        val_loss = 0.0

        # Confusion matrix, classification report
        y_pred = {  'Skill' : [], 'Turner1': [], 'Turner2': [], 'Type' : [], }
        y_true = {  'Skill' : [], 'Turner1': [], 'Turner2': [], 'Type' : [], }
        target_names = {
            'Skill' : ['jump', 'return from power', 'pushup', 'frog', 'other'],
            'Turner' : ['normal', 'crouger', 'cross', 'cross BW', 'jump over cross BW', 'EB', 'toad', 'toad BW', 'EB toad', 'TS', 'inverse toad', 'elephant', 'crougercross', 'pinwheel', 'suicide', 'inverse crouger', 'flip', 'T-toad', 'MULTIPLE+1', 'EB toad BW', 'L2-power-gym', 'L3-power-gym', 'L4-power-gym', 'UNKNOWN', 'jump-through', 'EB inverse toad'],
            'Type' : ['Double Dutch', 'Single Dutch', 'Irish Dutch', 'Chinese Wheel', 'Transition', 'Snapperlike'],
        }

        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader):
                with torch.amp.autocast(device_type='cuda'):
                    optimizer.zero_grad()  # Clear gradients
                    outputs = model(batch_X / 255)
                    
                    # Loss
                    total_batch_loss = self.__compute_losses(outputs=outputs, batch_y=batch_y, loss_fns=loss_fns)
                    val_loss += total_batch_loss.item()

                    # Accuracy
                    for key, pred in outputs.items():
                        target = batch_y[key]

                        if key in ['Skill', 'Turner1', 'Turner2', 'Type']:
                            pred = F.softmax(pred, dim=1)
                            max_scores, max_idx_class = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label

                            y_pred[key].extend(max_idx_class.data.cpu().numpy())
                            y_true[key].extend(target.data.cpu().numpy())
        
        print(f"="*80)
        classification_reports = {}
        for key in y_true.keys():
            # y_true_key = [int(i) for i in y_true[key]]
            # y_pred_key = [int(i) for i in y_pred[key]]
            classKey = key if key not in ['Turner1', 'Turner2'] else 'Turner'
            classification_reports_string = classification_report(y_true[key], y_pred[key], labels=range(len(target_names[classKey])), target_names=target_names[classKey], zero_division=0)
            classification_reports[key] = classification_report(y_true[key], y_pred[key], output_dict=True, labels=range(len(target_names[classKey])), target_names=target_names[classKey], zero_division=0)
            print(f"----- Details ----")
            print(classification_reports_string)
            print(f"="*80)

        print(f"Total (macro avg) accuracy", classification_reports['Skill']['macro avg'])

        return val_loss / len(dataloader), classification_reports['Skill']['macro avg'], classification_reports

    def train(self, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, trainparams: dict= {}, learning_rate=1e-5):
        try:
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            path = os.path.join(MODELWEIGHT_PATH, f"{modelname}.state_dict.pt")
            checkpointPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}.checkpoint.pt")

            
            DIM = 224
            repo = DataRepository()
            model = PYTORCH_MODELS_SKILLS[modelname](modelinfo=trainparams, df_table_counts=repo.get_skill_category_counts()).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            epoch_start = 0
            accuracies = {}
            if not from_scratch and os.path.exists(checkpointPath):
                checkpoint = torch.load(checkpointPath, weights_only=True)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_start = checkpoint['epoch'] + 1
                accuracies = {} if 'accuracies' not in checkpoint.keys() else checkpoint['accuracies']

            if unfreeze_all_layers:
                for param in model.parameters():
                    param.requires_grad = True

            train_generator = DataGeneratorSkills(
                frameloader=FrameLoader(repo),
                train_test_val="train",
                dim=(DIM,DIM),
                timesteps=trainparams['timesteps'],
                batch_size=trainparams['batch_size'],
            )
            val_generator = DataGeneratorSkills(
                frameloader=FrameLoader(repo),
                train_test_val="val",
                dim=(DIM,DIM),
                timesteps=trainparams['timesteps'],
                batch_size=trainparams['batch_size'],
            )
        
            dataloaderTrain = DataLoader(train_generator, batch_size=1, shuffle=True)
            dataloaderVal = DataLoader(val_generator, batch_size=1, shuffle=True)

            loss_fns = {
                'categorical': torch.nn.CrossEntropyLoss(),  # For outputs like 'Skill', 'Turner1'
                'regression': torch.nn.MSELoss()             # For scalar outputs
            }

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
                    # scaler.unscale_(optimizer)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Adjust as needed
                        optimizer.step()
                    
                    total_loss += total_batch_loss.item()
                    i+=1

                print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloaderTrain):.4f}")

                val_loss, macro_avg_accuracy, class_reports = self.validate(model=model, dataloader=dataloaderVal, optimizer=optimizer, loss_fns=loss_fns)
                accuracies[epoch] = macro_avg_accuracy
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f} (val loss = {val_loss})")

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                    'accuracies': accuracies,
                    'class_reports' : class_reports
                }, checkpointPath)
            
            # End training
            print(accuracies)
            repo = DataRepository()
            torch.save(model.state_dict(), path)

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()


    def predict(modelname, videoId):
        pass

    def __addPytorchTop(model):
        """Returns a given pytorch model with the skill top predictions"""
        raise NotImplementedError()

    def __addKerasTop(model):
        """Returns a given keras model with the skill top predictions"""
        raise NotImplementedError()