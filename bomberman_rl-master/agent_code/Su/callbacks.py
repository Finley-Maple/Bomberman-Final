import os
import pickle
import random

import math

import numpy as np
import torch

from agent_code.Su.model import DQN



PARAMETERS = 'save after 200 iterations'

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.policy_net = DQN(578, len(ACTIONS))
    if self.train :
        self.logger.info("Setting up model from scratch.")
    else:
        self.steps_done = 0
        self.logger.info("Loading model from saved state.")
        self.policy_net.load_state_dict(torch.load(f'network_parameters\{PARAMETERS}.pt'))
        self.policy_net.eval()
       
                   
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # TODO Exploration vs exploitation

    sample = random.random()
    
    self.steps_done += 1
    if self.train:
        self.exploration_rate = self.exploration_rate * (0.995) if self.exploration_rate > \
                    0.05 else 0.05 
        if sample > self.exploration_rate:
            with torch.no_grad():
                features = state_to_features(game_state)
                features_tensor = torch.from_numpy(features).float()
                action = self.policy_net(features_tensor)
                return ACTIONS[torch.argmax(action)]
        else:
            self.logger.debug("Choosing action purely at random.")
            self.logger.debug(f'Exploration rate: {self.exploration_rate}')
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
    else:
        features = state_to_features(game_state)
        features_tensor = torch.from_numpy(features).float()
        action = self.policy_net(features_tensor)
        print(action)
        return ACTIONS[torch.argmax(action)]


def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array of the same shape of arena   
        the values represents the state of the tile
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    field_shape = game_state["field"].shape
    hybrid_matrix = np.zeros(field_shape + (2,), dtype=np.double)
    field_channel = game_state["field"]
    
    # Arena channel
    # FREE = 0
    # WALL = -1
    # CRATE = 1
    # FREE_COIN = 2 # if coins are contained in the crate, then the value is 3
    # CRATE_COIN = 3
    # FREE_BOMB0 = 4 # just placed
    # FREE_BOMB1 = 5
    # FREE_BOMB2 = 6
    # FREE_BOMB3 = 7 # almost exploding
    FREE_SELF  = 0.1
    FREE_OTHER = 0.2

    # first add coins to the arena
    coins = game_state['coins']
    # add the coin value to each position according to the coin position
    field_channel[coins] += 2
    
    # then add bombs to the arena
    # Bomb 
    
    # add the BOMB value to each position according to the bomb timer
    field_channel = field_channel + np.where(game_state["explosion_map"] == 4, 4, 0)
    field_channel = field_channel + np.where(game_state["explosion_map"] == 3, 5, 0)
    field_channel = field_channel + np.where(game_state["explosion_map"] == 2, 6, 0)
    field_channel = field_channel + np.where(game_state["explosion_map"] == 1, 7, 0)

    # then add the agent to the arena
    _, _, _, (x, y) = game_state['self']
    field_channel[x, y] = FREE_SELF
    other_pos = [xy for (n, s, b, xy) in game_state['others']]
    field_channel[other_pos] = FREE_OTHER


    coin_incrate_channel = np.zeros(field_shape)
    field = game_state["field"]

    self_position = game_state["self"][3]
    crate_positions = np.argwhere(field == 1)
    for crate_position in crate_positions:
        distance = np.linalg.norm(crate_position - self_position)  # 计算距离
        inverse_distance = 1.0 / (distance + 1)  # 避免除以零
    coin_incrate_channel[tuple(crate_position)] = inverse_distance
    
    hybrid_matrix[..., 0] = field_channel
    hybrid_matrix[..., 1] = coin_incrate_channel
    



    return hybrid_matrix.reshape(-1)




    