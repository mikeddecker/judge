from constants import PYTORCH_MODELS_SKILLS
from managers.DataRepository import DataRepository
from managers.DataGeneratorSkillsTorch import DataGeneratorSkills
from managers.FrameLoader import FrameLoader
import gc
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cudnn.benchmark = True
scaler = torch.GradScaler()

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
                max_scores, max_idx_class = pred.max(dim=1)  # [B, n_classes] -> [B], # get values & indices with the max vals in the dim with scores for each class/label
                # print(max_scores)
                # print(max_idx_class)

                acc = (max_idx_class == target).sum().item() / max_scores.size(0)
                # print(f"Acc for {key} is {acc}")
                
                correctCounts[key] += acc
            # else:  # Regression
            #     loss = loss_fns['regression'](pred.squeeze(), target)
            
            # losses.append(loss)
        
        # Total loss (sum of all individual losses)
        return correctCounts

    def validate(self, model, dataloader, optimizer, loss_fns, device='cuda'):
        model.eval()
        val_loss = 0.0
        accuracyCounts = {
            'Skill' : 0, 
            'Turner1': 0,
            'Turner2': 0, 
            'Type' : 0,
        }
        with torch.no_grad():
            i = 0
            for batch_X, batch_y in tqdm(dataloader):
                with torch.amp.autocast(device_type='cuda'):
                    optimizer.zero_grad()  # Clear gradients
                    
                    # Forward pass
                    outputs = model(batch_X / 255)
                    total_batch_loss = self.__compute_losses(outputs=outputs, batch_y=batch_y, loss_fns=loss_fns)
                    accuracies = self.__compute_accuracy(outputs=outputs, targets=batch_y)
                    for key, value in accuracies.items():
                        accuracyCounts[key] += value
                    
                    val_loss += total_batch_loss.item()

                i+=1
            # print(f"Validation loss: {val_loss / i:.4f}")
        print(f"Validation accuracy", accuracyCounts)


        return val_loss / len(dataloader)

    def train(self, modelname, from_scratch, epochs, save_anyway, unfreeze_all_layers=False, trainparams: dict= {}):
        try:

            if modelname not in PYTORCH_MODELS_SKILLS.keys():
                raise ValueError(modelname)
            
            DIM = 224
            repo = DataRepository()
            model = PYTORCH_MODELS_SKILLS[modelname](modelinfo=trainparams, df_table_counts=repo.get_skill_category_counts()).to(device)

            repo = DataRepository()
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
        
            # Create DataLoader
            dataloaderTrain = DataLoader(train_generator, batch_size=1, shuffle=True)
            dataloaderVal = DataLoader(val_generator, batch_size=1, shuffle=True)

            # Define optimizer and loss functions
            optimizer = optim.Adam(model.parameters(), lr=1e-4)
            loss_fns = {
                'categorical': torch.nn.CrossEntropyLoss(),  # For outputs like 'Skill', 'Turner1'
                'regression': torch.nn.MSELoss()             # For scalar outputs
            }

            # Training loop
            for epoch in range(epochs):
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
                    if i % 100 == 20:
                        print(f"Loss: {total_loss / i:.4f}")

                print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloaderTrain):.4f}")

                val_loss = self.validate(model=model, dataloader=dataloaderVal, optimizer=optimizer, loss_fns=loss_fns)
                print(f"Epoch {epoch+1}, Validation Loss: {val_loss / len(dataloaderVal):.4f} (val loss = {val_loss})")
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