from typing import List
import torch
from torch import nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=128, num_layers=3, dropout_prob=0.2, device='cpu'):
        super(LinearModel, self).__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_prob))
        
        layers.append(nn.Linear(hidden_dim, output_size))
        
        self.model = nn.Sequential(*layers).to(device)
    
    def forward(self, x):
        return self.model(x.view(x.shape[0], -1))

class OneHotCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(OneHotCrossEntropyLoss, self).__init__()

    def forward(self, input_logits, target_one_hot):
        epsilon = 1e-8
        input_softmax_clipped = torch.clamp(input_logits, epsilon, 1.0 - epsilon)
        log_probs = torch.log(input_softmax_clipped)
        loss = -torch.sum(target_one_hot * log_probs, dim=1)
        loss = torch.mean(loss)
        return loss
