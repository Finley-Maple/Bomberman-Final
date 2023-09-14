import os
import random

import numpy as np
import torch
import os
from agent_code.zombie.TeacherFeatures import state_to_teacher_features

from agent_code.zombie.TeacherModel import DQN



from .Model import Maverick
from .ManagerFeatures import state_to_features


 
PARAMETERS = 'last_save' 
TEACHERPARAMETER='teacher_parameters'

ACTIONS = ['LEFT', 'RIGHT', 'UP', 'DOWN', 'WAIT', 'BOMB']


def setup(self):
    """
    This is called once when loading each agent.
    Preperation such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.network = Maverick()
    #############################################################
    self.teacher = DQN()
    filename = os.path.join("network_parameters", f'{TEACHERPARAMETER}.pt')
    self.teacher.load_state_dict(torch.load(filename))
    self.teacher.eval()
    ############################################################
    self.coinlist=[]
    self.bomb_buffer = 0


    if self.train:
        self.logger.info("Trainiere ein neues Model.")
        

    else:
        self.logger.info(f"Lade Model '{PARAMETERS}'.")
        filename = os.path.join("network_parameters", f'{PARAMETERS}.pt')
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()
        

    
    
    

def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.S
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if game_state is None:
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    #################################################################
    features = state_to_features(self, game_state)
    teacher_features = state_to_teacher_features(self, game_state)
    ################################################################
    Q = self.network(features)
    Q_teacher = self.teacher(teacher_features)
    ################################################################
    action_prob	= np.array(torch.softmax(Q,dim=1).detach().squeeze())
    action_prob_teacher	= np.array(torch.softmax(Q_teacher,dim=1).detach().squeeze())
    #################################################################
    best_action = ACTIONS[np.argmax(action_prob)]
    best_action_teacher = ACTIONS[np.argmax(action_prob_teacher)]


    if self.train: # Exploration vs exploitation
        eps = self.epsilon_arr[self.episode_counter]
        if random.random() <  eps:
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])

                   
    self.logger.info(f"Waehle Aktion {best_action} nach dem Hardmax der Q-Funktion")

    return best_action_teacher
    return best_action