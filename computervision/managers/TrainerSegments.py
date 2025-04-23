import gc
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from constants import PYTORCH_MODELS_SKILLS
from dotenv import load_dotenv
from managers.DataRepository import DataRepository
from managers.DataGeneratorSegmentationTorch import DataGeneratorSegmentation
from managers.FrameLoader import FrameLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

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

class TrainerSegments:
    def validate(self, model, dataloader, optimizer, loss_fns, device='cuda'):
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_X, batch_y in tqdm(dataloader):
                with torch.amp.autocast(device_type='cuda'):
                    optimizer.zero_grad()
                    outputs = model(batch_X / 255)
                    
                    # Loss
                    total_batch_loss = loss_fns['regression'](outputs, batch_y)
                    val_loss += total_batch_loss.item()

                    # Accuracy TODO : think about how
        
        return val_loss / len(dataloader)

    def train(self, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, trainparams: dict= {}, learning_rate=1e-5):
        try:
            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            path = os.path.join(MODELWEIGHT_PATH, f"{modelname}_segmentation.state_dict.pt")
            checkpointPath = os.path.join(MODELWEIGHT_PATH, f"{modelname}_segmentation.checkpoint.pt")

            
            DIM = 224
            repo = DataRepository()
            model = PYTORCH_MODELS_SKILLS[modelname](skill_or_segment="segments", modelinfo=trainparams, df_table_counts=repo.get_skill_category_counts()).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1)

            epoch_start = 0
            losses = []
            # accuracies = {}
            if not from_scratch and os.path.exists(checkpointPath):
                checkpoint = torch.load(checkpointPath)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                epoch_start = checkpoint['epoch'] + 1
                losses = checkpoint['losses']

            if unfreeze_all_layers:
                for param in model.parameters():
                    param.requires_grad = True

            train_generator = DataGeneratorSegmentation(
                frameloader=FrameLoader(repo),
                train_test_val="train",
                dim=(DIM,DIM),
                timesteps=trainparams['timesteps'],
                batch_size=trainparams['batch_size'],
            )
            val_generator = DataGeneratorSegmentation(
                frameloader=FrameLoader(repo),
                train_test_val="val",
                dim=(DIM,DIM),
                timesteps=trainparams['timesteps'],
                batch_size=trainparams['batch_size'],
            )
        
            dataloaderTrain = DataLoader(train_generator, batch_size=1, shuffle=True)
            dataloaderVal = DataLoader(val_generator, batch_size=1, shuffle=True)

            loss_fns = {
                'categorical': torch.nn.CrossEntropyLoss(),
                'regression': torch.nn.MSELoss()
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
                        batch_loss = loss_fns['regression'](outputs, batch_y)
                        batch_loss.backward()
                        optimizer.step()
                    
                    total_loss += batch_loss.item()
                    i+=1

                print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloaderTrain):.4f}")

                val_loss = self.validate(model=model, dataloader=dataloaderVal, optimizer=optimizer, loss_fns=loss_fns)
                losses.append(val_loss)
                scheduler.step(val_loss)
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}")

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
                    }, checkpointPath)
            
                    torch.save(model.state_dict(), path)

        except Exception as e:
            raise e
        finally:
            torch.cuda.empty_cache()
            gc.collect()
