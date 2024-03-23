import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset

from audio_toolbox.trainer import ModelTrainer
from audio_toolbox.models import CNNModel


if __name__ == '__main__':
    
    train_data = torch.load('processed_data/dl_data/mlp_data/pca_data/dl_modeling_train_data.pt')
    val_data = torch.load('processed_data/dl_data/mlp_data/pca_data/dl_modeling_val_data.pt')
    test_data = torch.load('processed_data/dl_data/mlp_data/pca_data/dl_modeling_test_data.pt')
    train_label = torch.load('processed_data/dl_data/mlp_data/pca_data/dl_modeling_train_label.pt')
    val_label = torch.load('processed_data/dl_data/mlp_data/pca_data/dl_modeling_val_label.pt')
    test_label = torch.load('processed_data/dl_data/mlp_data/pca_data/dl_modeling_test_label.pt')
    
    datasets = {
        'train': TensorDataset(train_data, train_label),
        'val': TensorDataset(val_data, val_label),
        'test': TensorDataset(test_data, test_label)
    }
    
    input_size = train_data.size(1)
    output_size = 10
    batch_size = 32

    model = CNNModel()
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-3  # Adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = ModelTrainer(datasets, model, loss_fn, optimizer, scheduler)
    trainer_config = {
        'save': False,
        'num_epochs': 100,
        'batch_size': batch_size
    }
    
    trainer.train(**trainer_config)