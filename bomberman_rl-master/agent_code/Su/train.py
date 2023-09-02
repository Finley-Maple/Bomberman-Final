import pickle
from collections import namedtuple, deque
from typing import List
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

import events as e
from agent_code.Su.model import DQN
from agent_code.Su.callbacks import state_to_features, ACTIONS
from agent_code.Su.replay_memory import ReplayMemory


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
EXPLORATION_RATE = 0.95

BATCH_SIZE = 128
GAMMA = 0.99

TARGET_UPDATE = 4
# Events



def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    
    self.policy_net = DQN(578, len(ACTIONS))
    self.target_net = DQN(578, len(ACTIONS))
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    

    self.exploration_rate = 1.0

    self.steps_done = 0
    self.rounds = 0

    self.optimizer = optim.RMSprop(self.policy_net.parameters())
    self.memory = ReplayMemory(100000)




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

    if old_game_state is not None:
        self.memory.push(state_to_features(old_game_state), 
                         [ACTIONS.index(self_action)],
                         state_to_features(new_game_state),
                         reward_from_events(self, events))

        optimize_model(self)

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state),reward_from_events(self, events)))



def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.

    Args:
        events:
        last_action:
        last_game_state:
    """
    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))



    # Update the target network, copying all weights and biases in DQN
    if self.rounds % TARGET_UPDATE == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done = 0
    

    self.rounds += 1

    if self.rounds % 200 == 0: #save parameters 200 times
        save_parameters(self, f"save after {self.rounds} iterations")
    
    



def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -3,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -250,
        e.GOT_KILLED: -10,
        e.CRATE_DESTROYED: 50
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


def optimize_model(self):
    #检查经验回放缓冲区（self.memory）中是否有足够的样本来执行优化。如果缓冲区中的样本数量小于BATCH_SIZE + 1，则不执行优化.
    if len(self.memory) < BATCH_SIZE + 1:
        return
    #从经验回放缓冲区中随机采样一个大小为BATCH_SIZE的批次，以用于训练。
    transitions = self.memory.sample(BATCH_SIZE)

    #经验回放缓冲区中采样的批次数据重新组织为一组状态、动作、下一个状态和奖励的序列，以便于后续的计算和训练。
    batch = Transition(*zip(*transitions))

    #创建一个布尔掩码，用于标识非最终状态的批次元素。
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state)), dtype=torch.bool)
    #将样本批次拆分为状态、下一个状态、奖励和动作。
    # 合并多个NumPy数组成一个单一的NumPy数组 # 转换为PyTorch张量
    state_batch = np.array(batch.state)
    state_batch = torch.tensor(state_batch).float()

    #用 NumPy 的 asarray 函数将 batch.action 转换为 NumPy 数组。
    action_batch = torch.tensor(np.asarray(batch.action, dtype=np.int64))

    next_state_batch = np.array(batch.next_state)
    next_state_batch = torch.tensor(next_state_batch).float()

    reward_batch = np.array(batch.reward)
    reward_batch = torch.tensor(reward_batch).float()

    #收集非最终下一个状态的张量。
    non_final_next_states = torch.cat([s for s in next_state_batch if s is not None])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    #计算当前状态下采取的动作的Q值
    state_action_values = self.policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    #计算下一个状态的V值，即期望的最大Q值。
    next_state_values = torch.zeros(BATCH_SIZE)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states.reshape(-1, 578)).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    for param in self.policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    self.optimizer.step()
    
def save_parameters(self, string):
    torch.save(self.policy_net.state_dict(), f"network_parameters/{string}.pt")

