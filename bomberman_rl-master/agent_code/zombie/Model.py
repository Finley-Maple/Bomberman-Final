
import torch.nn as nn
import torch.nn.functional as F

class Maverick(nn.Module):
    '''
    NN with one hidden layer to calculate Q-values 
    for 6 actions depending on features (game state)
    '''
    def __init__(self):
        super(Maverick, self).__init__()

        self.input_size = 579
        self.number_of_actions = 6

        #LAYERS
        self.layer1 = nn.Linear(in_features=self.input_size, out_features=256)
        self.layer2 = nn.Linear(in_features=256, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=self.number_of_actions)


    def forward(self, x):
        x = F.relu((self.layer1(x)))
        x = F.relu((self.layer2(x)))
        out = self.out(x)
        return out

    def initialize_training(self, 
                alpha,
                gamma, 
                epsilon, 
                buffer_size,
                batch_size, 
                loss_function,
                optimizer,
                training_episodes):
        self.gamma = gamma
        self.epsilon_begin = epsilon[0]
        self.epsilon_end = epsilon[1]
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.loss_function = loss_function
        self.training_episodes = training_episodes










    