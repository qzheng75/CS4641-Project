import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset

from audio_toolbox.trainer import ModelTrainer


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=32):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_size))

    def forward(self, x):
        # Assuming x is of shape (batch_size, input_size)
        return self.linear(x.view(x.shape[0], -1))

if __name__ == '__main__':
    
    train_data = torch.load('processed_data/dl_data_reduced/dl_modeling_train_data.pt')
    val_data = torch.load('processed_data/dl_data_reduced/dl_modeling_val_data.pt')
    test_data = torch.load('processed_data/dl_data_reduced/dl_modeling_test_data.pt')
    train_label = torch.load('processed_data/dl_data_reduced/dl_modeling_train_label.pt')
    val_label = torch.load('processed_data/dl_data_reduced/dl_modeling_val_label.pt')
    test_label = torch.load('processed_data/dl_data_reduced/dl_modeling_test_label.pt')
    
    datasets = {
        'train': TensorDataset(train_data, train_label),
        'val': TensorDataset(val_data, val_label),
        'test': TensorDataset(test_data, test_label)
    }
    
    input_size = train_data.size(1)
    output_size = 10
    batch_size = 32

    model = SimpleLinearModel(input_size, output_size)
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-5  # Adjust the learning rate as needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trainer = ModelTrainer(datasets, model, loss_fn, optimizer, scheduler)
    trainer_config = {
        'save': True,
        'num_epochs': 20,
        'batch_size': batch_size
    }
    
    trainer.train(**trainer_config)