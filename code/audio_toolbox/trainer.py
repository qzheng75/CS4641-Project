import os
import logging
import copy
from time import time
from datetime import datetime

from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
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
        self.metric = {'train': [], 'val': [], 'test': []}
    
    def get_dataloaders(self, datasets: dict):
        return {
            "train_dataloader": DataLoader(datasets['train'],
                                           batch_size=self.batch_size, shuffle=True),
            "val_dataloader": DataLoader(datasets['val'],
                                batch_size=self.batch_size, shuffle=True),
            "test_dataloader": DataLoader(datasets['test'],
                                batch_size=self.batch_size, shuffle=True),
        }
        
    def train(self, **train_config):
        save = train_config.get('save', False)
        num_epochs = train_config.get('num_epochs', 20)
        batch_tqdm = train_config.get('batch_tqdm', False)
        log_per = train_config.get('log_per', 1)
        metric = train_config.get('metric', 'acc')
        
        if save:
            save_dir = train_config.get('save_dir', 'training_results')
            job_identifier = train_config.get('job_identifier', 'my_train_job')
            os.makedirs(save_dir, exist_ok=True)
            
            identifier = job_identifier + '@' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            save_dir = os.path.join(save_dir, identifier)
            os.makedirs(save_dir)
            
        min_loss_epoch = 0
        best_metric = torch.inf if metric == 'loss' else 0
        best_state_dict = None
        early_stopping = train_config.get('early_stopping', False)
        patience = train_config.get('patience', 5)
        no_improve_epochs = 0

        for epoch in range(num_epochs):
            batch_train_loss = []
            self.model.train()
            start_time = time()
            train_loader_iter = iter(self.dataloaders["train_dataloader"])
            pbar = tqdm(range(0, len(self.dataloaders["train_dataloader"])), disable=not batch_tqdm)
            for i in pbar:
                data, targets = next(train_loader_iter)
                out_logit = self.model(data)
                # out = torch.softmax(out_logit, dim=1)
                loss = self.loss_fn(out_logit, targets)
                batch_train_loss.append(loss.item())
                pbar.set_description(f"Batch {i} Loss {np.mean(batch_train_loss).item():.4f}")
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            if metric == 'loss':
                batch_train_metric = np.mean(batch_train_loss)
                batch_val_metric = self.validate(split='val')
                batch_test_metric = self.validate(split='test')
            elif metric == 'acc':
                batch_train_metric = self.get_acc_score(self.dataloaders['train_dataloader'])
                batch_val_metric = self.get_acc_score(self.dataloaders['val_dataloader'])
                batch_test_metric = self.get_acc_score(self.dataloaders['test_dataloader'])
                
            self.metric['train'].append(batch_train_metric)
            self.metric['val'].append(batch_val_metric)
            self.metric['test'].append(batch_test_metric)
            end_time = time()

            if log_per > 0 and epoch % log_per == 0:
                logging.info(f"Epoch {epoch+1:04d}, Learning rate: {self.optimizer.param_groups[0]['lr']:.6f}, "
                            f"Training metric: {batch_train_metric:.5f}, "
                            f"Val metric: {batch_val_metric:.5f}, "
                            f"Test metric: {batch_test_metric:.5f}, "
                            f"Epoch time: {end_time - start_time:.5f}")      
                
            if (metric == 'loss' and batch_val_metric < best_metric) or (metric == 'acc' and batch_val_metric > best_metric):
                best_metric = batch_val_metric
                no_improve_epochs = 0
                min_loss_epoch = epoch
                best_state_dict = self.model.state_dict()
            else:
                no_improve_epochs += 1

            if early_stopping and no_improve_epochs >= patience:
                logging.info("Early stopping triggered")
                break

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        logging.info(f"Best model at epoch {min_loss_epoch+1:04d} with validation metric {best_metric:.5f}")

        if save:
            logging.info(f'Saving the best model state_dict to {identifier}')
            save_state_dict_folder = os.path.join(save_dir, 'checkpoints')
            os.makedirs(save_state_dict_folder)
            save_state_dict_path = os.path.join(save_state_dict_folder, 'best_model.pt')
            torch.save(best_state_dict, save_state_dict_path)   
        return best_state_dict
    
    def validate(self, split):
        self.model.eval()
        batch_loss = []
        for _, (data, targets) in enumerate(self.dataloaders[f'{split}_dataloader']):
            with torch.no_grad():
                out = self.model(data)#.softmax(dim=1)
                loss = self.loss_fn(out, targets)
            batch_loss.append(loss.item())
        return np.mean(batch_loss)
    
    def get_metric_record(self):
        return self.metric
    
    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            out = self.model(X).softmax(dim=1).argmax(dim=1)
        return out
    
    def get_acc_score(self, dataloader):
        acc = []
        self.model.eval()
        for _, (data, targets) in enumerate(dataloader):
            with torch.no_grad():
                out = self.model(data).softmax(dim=1).argmax(dim=1)
                acc.append(accuracy_score(targets.cpu().numpy(), out.cpu().numpy()))
        return np.mean(acc)
            
    def load_model(self, state_dict):
        self.model.load_state_dict(state_dict)