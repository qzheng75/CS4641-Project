from typing import List
import torch
from torch import nn
import torch.nn.functional as F


class SimpleLinearModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim=64, dropout_prob=0., device='cpu'):
        super(SimpleLinearModel, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_size, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_prob),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Dropout(dropout_prob),
                                    nn.Linear(hidden_dim, output_size)).to(device)

    def forward(self, x):
        # Assuming x is of shape (batch_size, input_size)
        return self.linear(x.view(x.shape[0], -1))

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
