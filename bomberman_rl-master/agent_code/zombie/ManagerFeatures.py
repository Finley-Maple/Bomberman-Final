import torch
import numpy as np
from collections import deque
import heapq
import random
import bisect


def state_to_features(self, game_state: dict) -> torch.tensor:
    
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
    
    for i in game_state["others"]:
        arena_channel[i[3]] = 2

    arena_channel[game_state["self"][3]] = 3

    for coin in game_state['coins']:
        arena_channel[coin] = 5
        # 检查坐标是否已经在self.coinlist中
        if coin not in self.coinlist:
            # 如果不在列表中，则将其添加到self.coinlist
            self.coinlist.append(coin)
            # add the coin value to each position according to the coin position

    field_shape = game_state["field"].shape
    bomb_channel = np.zeros(field_shape)
    for i in game_state["bombs"]:
        bomb_channel[i[0]] = i[1]


    coin_channel = np.array([len(self.coinlist)])

    arena_channel =arena_channel.reshape(-1)
    bomb_channel=bomb_channel.reshape(-1)
    coin_channel=coin_channel.reshape(-1)
    features = np.concatenate([arena_channel, bomb_channel, coin_channel])
    features_tensor = torch.tensor(features, dtype=torch.float32)
    # 添加一个维度
    features_tensor = features_tensor.unsqueeze(0)

    # 返回形状为 torch.Size([1, 579]) 的特征张量  
    
    return  features_tensor