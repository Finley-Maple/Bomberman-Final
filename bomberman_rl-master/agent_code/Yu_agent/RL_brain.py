import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import settings as s

# Hyper parameters -- DO modify
fc1_in = 23
fc2_in = 32
lr = 0.001 # learning rate

# Use policy net to choose action
class PolicyNet(nn.Module):
    def __init__(self, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(fc1_in, fc2_in)
        self.fc2 = nn.Linear(fc2_in, n_actions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        # nparray to tensor
        x = torch.from_numpy(x).float()
        # add the first dimension for those with batch
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(x.view(-1, fc1_in)))
        x = self.fc2(x)
        # remove the first dimension
        if x.shape[0] == 1:
            x = x.squeeze(0)    
        return x

    def loss_function(self, State, Action, Reward):
        '''
        State: nparray
        Action: nparray
        Reward: nparray
        '''
        Q = self.forward(State)
        # if Action is nparray, convert it to tensor
        if type(Action) == np.ndarray:
            Action = torch.from_numpy(Action).long()
        # if reward is nparray, convert it to tensor
        if type(Reward) == np.ndarray:
            Reward = torch.from_numpy(Reward).float()
        # calculate the negative log probability
        neg_log_prob = self.loss_fn(Q, Action)
        return torch.sum(neg_log_prob*Reward)
    