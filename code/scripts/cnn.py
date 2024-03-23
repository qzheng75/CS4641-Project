import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn

from audio_toolbox.dataset import AudioOTFDataset
from audio_toolbox.models import CNNModel, SimpleCNN
from audio_toolbox.trainer import ModelTrainer


if __name__ == '__main__':
    train_data = torch.load('processed_data/dl_data/cnn_data/dl_modeling_train_data.pt')
    val_data = torch.load('processed_data/dl_data/cnn_data/dl_modeling_val_data.pt')
    test_data = torch.load('processed_data/dl_data/cnn_data/dl_modeling_test_data.pt')

    train_label = torch.load('processed_data/dl_data/cnn_data/dl_modeling_train_label.pt')
    val_label = torch.load('processed_data/dl_data/cnn_data/dl_modeling_val_label.pt')
    test_label = torch.load('processed_data/dl_data/cnn_data/dl_modeling_test_label.pt')
    
    datasets = {
        'train': TensorDataset(train_data, train_label),
        'val': TensorDataset(val_data, val_label),
        'test': TensorDataset(test_data, test_label)
    }

    input_size = train_data.size(1)
    output_size = 10
    batch_size = 16

    num_conv_layers = 2
    in_channels = [6, 16]
    out_channels = [16, 32]
    channel_widths = [8, 8]
    num_post_cnn_fc_layers = 2
    linear_in_dims = [32, 64]
    linear_out_dims = [64, 10]
    
    model_config = {
        "num_conv_layers": num_conv_layers,
        "in_channels": in_channels,
        "out_channels": out_channels,
        "channel_widths": channel_widths,
        "num_post_cnn_fc_layers": num_post_cnn_fc_layers,
        "linear_in_dims": linear_in_dims,
        "linear_out_dims": linear_out_dims
    }
    # model = CNNModel(**model_config)
    model = SimpleCNN(1290)

    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-4  # Adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    trainer = ModelTrainer(datasets, model, loss_fn, optimizer, scheduler)
    trainer_config = {
        'save': False,
        'num_epochs': 100
    }
    trainer.train(**trainer_config)