from collections import namedtuple, deque

from typing import List
import os
import events as e
from .callbacks import state_to_features, ACTIONS, MODEL_NAME
from .RL_brain import PolicyNet
import settings as s
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
import matplotlib as mpl

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
gamma = 0.95 # discount factor
# channels = 3

PERIOD = 500

IMITATION_EPISODES = 500

# Events
# Defined by ICEC 2019
LAST_MAN_STANDING = "LAST_MAN_STANDING"
CLOSER_TO_ENEMY = "CLOSER_TO_ENEMY"
CLOSEST_TO_ENEMY = "CLOSEST_TO_ENEMY"
FARTHER_TO_ENEMY = "FARTHER_TO_ENEMY"
DANGER_ZONE_BOMB = "DANGER_ZONE_BOMB"
SAFE_CELL_BOMB = "SAFE_CELL_BOMB"
ALREADY_VISITED_EVENT = "ALREADY_VISITED_EVENT"
CLOSER_TO_COIN = "CLOSER_TO_COIN"
CLOSEST_TO_COIN = "CLOSEST_TO_COIN"

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.action_space = [i for i in range(len(ACTIONS))]
    self.gamma = gamma
    self.state_memory = []
    self.action_memory = []
    self.reward_memory = []
    self.reward_history = []
    self.loss_history = []
    
    self.visited = np.zeros((17, 17))
    self.visited_before = np.zeros((17, 17))

    self.game_score = 0
    self.game_score_arr = []
    self.other_game_score = 0
    self.other_game_score_arr = []
    self.win = 0
    self.win_arr = []
    self.total_episodes = 0

    if not os.path.exists('./figures'):
        os.makedirs('./figures')


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    When this method is called, self.events will contain a list of all game
    events relevant to your agent that occurred during the previous step. Consult
    settings.py to see what events are tracked. You can hand out rewards to your
    agent based on these events and your knowledge of the (new) game state.

    This is *one* of the places where you could update your agent.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

    # Distances to enemies
    pos_current = new_game_state["self"][3]
    
    if len(new_game_state["others"]) > 0:
        for _, _, _, pos_enemy in new_game_state["others"]:
            dist = np.linalg.norm(np.array(pos_current) - np.array(pos_enemy))
            if dist < 4:
                events.append(CLOSER_TO_ENEMY)
                self.logger.debug(f'Add game event {CLOSER_TO_ENEMY} in step {new_game_state["step"]}')
            if dist == 1:
                events.append(CLOSEST_TO_ENEMY)
                self.logger.debug(f'Add game event {CLOSEST_TO_ENEMY} in step {new_game_state["step"]}')
            if dist > 4:
                events.append(FARTHER_TO_ENEMY)
                self.logger.debug(f'Add game event {FARTHER_TO_ENEMY} in step {new_game_state["step"]}')

    # Distances to coins
    if len(new_game_state["coins"]) > 0:
        for pos_coin in new_game_state["coins"]:
            dist = np.linalg.norm(np.array(pos_current) - np.array(pos_coin))
            if dist < 4:
                events.append(CLOSER_TO_COIN)
                self.logger.debug(f'Add game event {CLOSER_TO_COIN} in step {new_game_state["step"]}')
            if dist == 1:
                events.append(CLOSEST_TO_COIN)
                self.logger.debug(f'Add game event {CLOSEST_TO_COIN} in step {new_game_state["step"]}')
    
    # Bomb blast range
    # TODO What happens if two bombs are in reach of current position?
    current_bombs = new_game_state["bombs"]
    is_getting_bombed = False
    for (x, y), countdown in current_bombs:
        for i in range(0, s.BOMB_POWER + 1):
            if new_game_state['field'][x + i, y] == -1:
                break
            # Check current position
            if pos_current == (x + i, y):
                is_getting_bombed = True
        for i in range(0, s.BOMB_POWER + 1):
            if new_game_state['field'][x - i, y] == -1:
                break
            # Check current position
            if pos_current == (x - i, y):
                is_getting_bombed = True
        for i in range(0, s.BOMB_POWER + 1):
            if new_game_state['field'][x, y + i] == -1:
                break
            # Check current position
            if pos_current == (x, y + i):
                is_getting_bombed = True
        for i in range(0, s.BOMB_POWER + 1):
            if new_game_state['field'][x, y - i] == -1:
                break
            # Check current position
            if pos_current == (x, y - i):
                is_getting_bombed = True

    if is_getting_bombed:
        events.append(DANGER_ZONE_BOMB)
        self.logger.debug(f'Add game event {DANGER_ZONE_BOMB} in step {new_game_state["step"]}')
    else:
        events.append(SAFE_CELL_BOMB)
        self.logger.debug(f'Add game event {SAFE_CELL_BOMB} in step {new_game_state["step"]}')

    if self.visited_before[pos_current[0]][pos_current[1]] == 1:
        events.append(ALREADY_VISITED_EVENT)
        self.logger.debug(f'Add game event {ALREADY_VISITED_EVENT} in step {new_game_state["step"]}')

    self.visited_before = self.visited
    self.visited[pos_current[0]][pos_current[1]] = 1

    store_transition(self, old_game_state, self_action, events, new_game_state)



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    
    self.total_episodes += 1

    # save parameters every PERIOD episodes
    if self.total_episodes % PERIOD == 0:
        torch.save(self.model.state_dict(), f"network_parameters/save after {self.total_episodes}.pt")

    store_transition(self, last_game_state, last_action, events, new_game_state=None)

    reset_params(self)

    track_game_result(self, last_game_state)

    # clear log every PERIOD episodes
    if self.total_episodes % PERIOD == 0:
        self.logger.handlers.clear()

    learn(self)

    # pytorch save
    torch.save(self.model.state_dict(), MODEL_NAME)

    # if episodes is larger than IMITATION_EPISODES, start training
    if self.total_episodes > IMITATION_EPISODES:
        self.rulebased = False

def store_transition(self, old_game_state, self_action, events, new_game_state):
    '''
    store the transition in the replay memory
    '''
    self.state_memory.append(state_to_features(self, old_game_state))
    # print(self_action)
    if self_action is None:
        self_action = "WAIT"
    action = ACTIONS.index(self_action)
    one_hot_action = np.zeros(len(ACTIONS))
    one_hot_action[action] = 1
    self.action_memory.append(action)
    reward = reward_from_events(self, events, new_game_state)
    self.reward_memory.append(reward)
    
def reset_params(self):
    '''
    reset the parameters of the agent
    '''
    self.visited = np.zeros((17, 17))
    self.visited_before = np.zeros((17, 17))

def reward_from_events(self, events: List[str], new_game_state = None) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.MOVED_RIGHT: -0.01,
        e.MOVED_LEFT: -0.01,
        e.MOVED_UP: -0.01,
        e.MOVED_DOWN: -0.01,
        e.WAITED: -0.03,
        e.INVALID_ACTION: -0.03,
        e.BOMB_DROPPED: -0.01,
        e.KILLED_SELF: -2,
        e.GOT_KILLED: -3,
        e.CRATE_DESTROYED: 0.5,
        CLOSER_TO_ENEMY: 0.002,
        CLOSEST_TO_ENEMY: 0.01,
        FARTHER_TO_ENEMY: -0.001,
        DANGER_ZONE_BOMB: -0.000666,
        SAFE_CELL_BOMB: 0.002,
        ALREADY_VISITED_EVENT: -0.0001,
        CLOSER_TO_COIN: 0.01,
        CLOSEST_TO_COIN: 0.3
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    
    if new_game_state is not None:
        reward_sum += count_destroyed_crates(events, new_game_state)
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def count_destroyed_crates(events, new_game_state):
    destroyed_count = 0
    if e.BOMB_DROPPED in events:
        field = new_game_state['field']
        bomb_pos=new_game_state["self"][3]
        rows, cols = field.shape
        x, y = bomb_pos
        
        # 定义四个方向的偏移量
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        
        # 遍历四个方向
        for dx, dy in directions:
            for i in range(1, 4):  # 炸弹向外扩散3个格子
                nx, ny = x + dx * i, y + dy * i
                
                # 检查是否越界
                if 0 <= nx < rows and 0 <= ny < cols:
                    tile = field[nx, ny]
                    
                    # 如果遇到墙，停止在该方向上的扩散
                    if tile == -1:
                        break
                    
                    # 如果遇到箱子，增加炸毁的箱子计数
                    elif tile == 1:
                        destroyed_count += 1
    
    return destroyed_count * 0.33

# policy based learning
def learn(self):
    state_memory = np.array(self.state_memory)
    action_memory = np.array(self.action_memory)
    reward_memory = np.array(self.reward_memory)

    reward = np.zeros_like(reward_memory)
    for t in range(len(reward_memory)):
        reward_sum = 0
        discount = 1
        for k in range(t, len(reward_memory)):
            reward_sum += reward_memory[k] * discount
            discount *= self.gamma
        reward[t] = reward_sum

    # record the reward 
    self.reward_history.append(reward[0])
    
    # for every PERIOD episodes, plot the reward
    if self.total_episodes % PERIOD == 0:
        track_reward(self)

    # normalize the rewards
    mean = np.mean(reward)
    std = np.std(reward) if np.std(reward) > 0 else 1
    reward = (reward-mean)/std
    # print(reward)
    # calculate the loss
    self.model.optimizer.zero_grad()
    loss = self.model.loss_function(state_memory, action_memory, reward)
    
    # track the loss
    self.loss_history.append(loss.item())
    
    # for every PERIOD episodes, plot the loss
    if self.total_episodes % PERIOD == 0:
        track_loss(self)

    # train the model with pytorch
    loss.backward()
    self.model.optimizer.step()

    # clean out the memory
    self.state_memory = []
    self.action_memory = []
    self.reward_memory = []

def track_game_result(self, last_game_state):

    

    _, self_score, _, _ = last_game_state['self']
    self.game_score = self_score
    self.game_score_arr.append(self.game_score)
    

    # track the highest score of the other agent
    for i in range(1, len(last_game_state['others'])+1):
        _, other_score, _, _ = last_game_state['others'][-i]
        if other_score > self.other_game_score:
            self.other_game_score = other_score
    
    self.other_game_score_arr.append(self.other_game_score)
    
    # record the win
    # if self.game_score > self.other_game_score:
    #     self.win += 1
    # self.win_arr.append(self.win)

    # every 50 episodes, plot the progress
    if self.total_episodes % PERIOD == 0:
        track_game_score(self, smooth=False)
        track_other_game_score(self, smooth=True)
        # track_win(self, smooth=True)

def track_game_score(self, smooth=False):
    '''
    Plot our gamescore -> helpful to see if our training is working without much time effort

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param smooth: calculate running mean if smooth==True, default: False
    '''
    
    self.game_score_arr.append(self.game_score)
    self.game_score = 0

    y = self.game_score_arr
    if smooth:
        window_size = self.total_episodes // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title('score during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('episode', fontsize=25, fontweight='bold')
    ax.set_ylabel('points', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    # ax.set_yticks(range(255)[::10])
    ax.set_yticks(range(255))
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)
    plt.savefig('./figures/self_scores.png')
    plt.close()

def track_other_game_score(self, smooth=False):
    '''
    Plot our gamescore -> helpful to see if our training is working without much time effort

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param smooth: calculate running mean if smooth==True, default: False
    '''
    
    self.other_game_score_arr.append(self.other_game_score)
    self.other_game_score = 0

    y = self.other_game_score_arr
    if smooth:
        window_size = self.total_episodes // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title('other score during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('episode', fontsize=25, fontweight='bold')
    ax.set_ylabel('points', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    # ax.set_yticks(range(255)[::10])
    ax.set_yticks(range(255))
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)
    plt.savefig('./figures/other_scores.png')
    plt.close()

def track_win(self, smooth=False):
    '''
    Plot our gamescore -> helpful to see if our training is working without much time effort

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param smooth: calculate running mean if smooth==True, default: False
    '''
    
    self.win_arr.append(self.win)
    self.win = 0

    y = self.win_arr
    if smooth:
        window_size = self.total_episodes // 25
        if window_size < 1:
            window_size = 1
        y = uniform_filter1d(y, window_size, mode="nearest", output="float")
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title('wins during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('episode', fontsize=25, fontweight='bold')
    ax.set_ylabel('wins', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    # ax.set_yticks(range(255)[::10])
    ax.set_yticks(range(255))
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)

    plt.savefig('./figures/wins.png')
    plt.close()

def track_loss(self):

    y = self.loss_history 
    x = range(len(y))
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('Loss during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=25, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    ax.tick_params(labelsize=16)

    ax.plot(x, y, color='blue', linewidth=0.5, alpha=0.7, zorder=0)


    plt.savefig('./figures/loss_progress.png')
    plt.close()

def track_reward(self):
    '''
    Plot our gamescore -> helpful to see if our training is working without much time effort

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param smooth: calculate running mean if smooth==True, default: False
    '''
    y = self.reward_history
    x = range(len(y))

    fig, ax = plt.subplots(figsize=(20,10))
    ax.set_title('reward during training', fontsize=35, fontweight='bold')
    ax.set_xlabel('episode', fontsize=25, fontweight='bold')
    ax.set_ylabel('points', fontsize=25, fontweight='bold')
    ax.grid(axis='y', alpha=0.2, color='gray', zorder=-1)
    # ax.set_yticks(range(255)[::10])
    ax.set_yticks(range(255))
    ax.tick_params(labelsize=16)

    ax.plot(x,y,color='gray',linewidth=0.5, alpha=0.7, zorder=0)

    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["red","darkorange","green"])
    ax.scatter(x,y,c=y,cmap=cmap,s=40, alpha=0.5, zorder=1)
    plt.savefig('figures/reward_progress.png')
    plt.close()