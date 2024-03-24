import os
import logging
import copy
from time import time
from datetime import datetime

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


logging.basicConfig(level=logging.INFO)

class ModelTrainer:
    def __init__(self,
                 datasets: dict,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 lr_scheduler: torch.optim.lr_scheduler._LRScheduler=None,
                 *args, **kwargs):
        self.batch_size = kwargs.get("batch_size", 32)
        
        assert all(key in datasets.keys() for key in ['train', 'val', 'test'])
        self.dataloaders = self.get_dataloaders(datasets)
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
    
    def get_dataloaders(self, datasets: dict):
        return {
            "train_dataloader": DataLoader(datasets['train'],
                                           batch_size=self.batch_size, shuffle=True),
            "val_dataloader": DataLoader(datasets['train'],
                                batch_size=self.batch_size, shuffle=True),
            "test_dataloader": DataLoader(datasets['train'],
                                batch_size=self.batch_size, shuffle=True),
        }
        
    def train(self, **train_config):
        save = train_config.get('save', False)
        num_epochs = train_config.get('num_epochs', 20)
        batch_tqdm = train_config.get('batch_tqdm', False)
        
        if save:
            save_dir = train_config.get('save_dir', 'training_results')
            job_identifier = train_config.get('job_identifier', 'my_train_job')
            os.makedirs(save_dir, exist_ok=True)
            
            identifier = job_identifier + '@' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_dir = os.path.join(save_dir, identifier)
            os.makedirs(save_dir)
            
        min_loss_epoch = 0
        min_loss = torch.inf
        best_state_dict = None
        
        for epoch in range(num_epochs):
            batch_train_loss = []
            start_time = time()
            for batch_idx, (data, targets) in tqdm(enumerate(self.dataloaders['train_dataloader']),
                                                    disable=not batch_tqdm,
                                                    desc=f'Epoch {epoch+1:04d}'):
                out_logit = self.model(data)
                out = torch.softmax(out_logit, dim=1)
                loss = self.loss_fn(out, targets)
                batch_train_loss.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            batch_train_loss = np.mean(batch_train_loss)
            batch_val_loss = self.validate(split='val')
            batch_test_loss = self.validate(split='test')
            end_time = time()

            logging.info(f"Epoch {epoch+1:04d}, Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}, "
                        f"Training loss: {batch_train_loss:.5f}, "
                        f"Val loss: {batch_val_loss:.5f}, "
                        f"Test loss: {batch_test_loss:.5f}, "
                        f"Epoch time: {end_time - start_time:.5f}")

            if batch_val_loss < min_loss:
                min_loss = batch_val_loss
                min_loss_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info(f"Best model at epoch {min_loss_epoch+1:04d} with validation loss {min_loss:.5f}")

        if save:
            logging.info(f'Saving the best model state_dict to {identifier}')
            save_state_dict_folder = os.path.join(save_dir, 'checkpoints')
            os.makedirs(save_state_dict_folder)
            save_state_dict_path = os.path.join(save_state_dict_folder, 'best_model.pt')
            torch.save(best_state_dict, save_state_dict_path)   
    
    def validate(self, split):
        self.model.eval()
        batch_loss = []
        for _, (data, targets) in enumerate(self.dataloaders[f'{split}_dataloader']):
            with torch.no_grad():
                out = self.model(data).softmax(dim=1)
                loss = self.loss_fn(out, targets)
            batch_loss.append(loss.item())
        return np.mean(batch_loss)
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out_logit = self.model(X)
        out_probs = torch.softmax(out_logit, dim=1)
        out_labels = torch.argmax(out_probs, dim=1)
        return out_labels
            