import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings as s

# Hyper parameters -- DO modify
conv1_in = 3
conv1_out = 8
# conv2_in = 4
# conv2_out = 8
fc1_in = 8*15*15
fc2_in = 32
kernel_size = 3
lr = 0.001 # learning rate

class DeepQNetwork(nn.Module):
    def __init__(self, n_actions):
        super(DeepQNetwork, self).__init__()
        self.model_sequence = nn.Sequential(
            nn.Conv2d(in_channels=conv1_in, out_channels=conv1_out, kernel_size=kernel_size),  # in_channels, out_channels, kernel size
            nn.ReLU(),
            # nn.Conv2d(in_channels=conv2_in, out_channels=conv2_out, kernel_size=kernel_size),
            # nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(fc1_in, fc2_in),
            nn.ReLU(),
            nn.Linear(fc2_in, n_actions)
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
            
    def forward(self, x):
        # add dimension for batch size
        x = self.model_sequence(x.view(-1, conv1_in, s.ROWS, s.COLS))
        # print(x.shape)
        return x
