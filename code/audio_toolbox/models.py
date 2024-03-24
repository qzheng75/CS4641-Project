from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64, device='cpu'):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, output_size)).to(device)

    def forward(self, x):
        # Assuming x is of shape (batch_size, input_size)
        return self.linear(x.view(x.shape[0], -1))
    

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, channel_widths, device=None):
        super(CNNBlock, self).__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=(5, 5),
                          stride=(3, 4),
                          padding=(2, 2)),
                # nn.BatchNorm2d(out_channels),
                nn.AvgPool2d(kernel_size=(5, 5),
                          stride=(3, 4),
                          padding=(2, 2)),
                nn.ReLU(),
            ).to(device)
        
    def forward(self, x):
        return self.model(x)

class CNNModel(nn.Module):
    def __init__(self,
                 num_conv_layers: int,
                 in_channels: List[int],
                 out_channels: List[int],
                 channel_widths: List[int],
                 num_post_cnn_fc_layers: int,
                 linear_in_dims: int,
                 linear_out_dims: int,
                 device=None):
        super(CNNModel, self).__init__()
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.conv_layers = nn.ModuleList()
        for i in range(num_conv_layers):
            self.conv_layers.append(CNNBlock(in_channels[i], out_channels[i],
                                             channel_widths[i], device))
        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        
        self.post_cnn_layers = nn.ModuleList()
        for i in range(num_post_cnn_fc_layers):
            self.post_cnn_layers.append(nn.Linear(linear_in_dims[i], linear_out_dims[i]).to(device))
        
        
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.global_pooling(x).view(x.shape[0], -1)
        for layer in self.post_cnn_layers:
            x = layer(x)
        return x
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(6, 16, kernel_size=(3, 3), padding=(1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AdaptiveMaxPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x