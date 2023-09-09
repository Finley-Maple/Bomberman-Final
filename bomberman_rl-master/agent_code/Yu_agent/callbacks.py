import os
import pickle
import random

import numpy as np
import torch
# Action space without the bomb action
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

MODEL_NAME = 'coin-model.pth'

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
    if self.train:
        if os.path.isfile(MODEL_NAME):
            self.logger.info("Loading model from saved state.")
            # pytorch load
            self.model = torch.load(MODEL_NAME)
        else:
            self.logger.info("Setting up model from scratch.")
            weights = np.random.rand(len(ACTIONS))
            self.model = weights / weights.sum()
    else:
        self.logger.info("Loading model from saved state.")
        # with open("my-saved-model.pt", "rb") as file:
        #     self.model = pickle.load(file)
        # pytorch load
        self.model = torch.load(MODEL_NAME)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    if self.train:
        if random.random() < self.epsilon:
            self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        else:
            features = state_to_features(game_state)
            features = torch.from_numpy(features).float()
            action = self.model.forward(features)
            # print(action)
            self.logger.debug("Querying model for action.")
            return ACTIONS[torch.argmax(action)]
    else:
        features = state_to_features(game_state)
        features = torch.from_numpy(features).float()
        action = self.model.forward(features)
        # print(action)
        self.logger.debug("Querying model for action.")
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

    arena_channel = game_state['field']
    
    # first: Arena channel
    # FREE = 0
    # WALL = -1
    # CRATE = 1
    # FREE_COIN = 2 # if coins are contained in the crate, then the value is 3
    
    # Second: Bomb channel
    bomb_channel = np.zeros(arena_channel.shape)
    # No_BOMB = 0
    # FREE_BOMB0 = 1 # just placed
    # FREE_BOMB1 = 2
    # FREE_BOMB2 = 3
    # FREE_BOMB3 = 4 # almost exploding

    # Third: Agent channel
    agent_channel = np.zeros(arena_channel.shape)
    FREE_SELF = 1
    FREE_OTHER = 2

    # first add coins to the arena
    coins = game_state['coins']
    # add the coin value to each position according to the coin position
    arena_channel[coins] += 2
    
    # construct the bomb channel
    bombs = game_state['bombs']
    bomb_pos = [xy for (xy, t) in bombs]
    bomb_timer = [t for (xy, t) in bombs]
    # check the type of bomb_pos and bomb_timer
    if not bomb_pos == []:
        # print(bomb_pos)
        # print(bomb_timer)
        # add the BOMB value to each position according to the bomb timer
        bomb_channel[np.array(bomb_pos)[np.array(bomb_timer) == 4]] += 3
        bomb_channel[np.array(bomb_pos)[np.array(bomb_timer) == 3]] += 4
        bomb_channel[np.array(bomb_pos)[np.array(bomb_timer) == 2]] += 5
        bomb_channel[np.array(bomb_pos)[np.array(bomb_timer) == 1]] += 6

    # construct the agent channel
    _, _, _, (x, y) = game_state['self']
    agent_channel[x, y] = FREE_SELF

    if not game_state['others'] == []:
        other_pos = [xy for (n, s, b, xy) in game_state['others']]
        agent_channel[other_pos] = FREE_OTHER
    
    # combine the channels in (3,arena_channel.shape)
    hybrid_matrix = np.array([arena_channel, bomb_channel, agent_channel])
    return hybrid_matrix
