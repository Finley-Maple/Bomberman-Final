from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from .RL_brain import DeepQNetwork
import settings as s
import numpy as np
import torch

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 3  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
gamma = 0.99 # discount factor
mem_size = 10000 # memory size
epsilon = 1.0 # exploration rate
batch_size = 32 # batch size
replace_target = 100 # replace target network after ... steps
q_next_dir = 'q_next'
q_eval_dir = 'q_eval'
channels = 3

MODEL_NAME = 'coin-model.pth'

# Action space without the bomb action
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.action_space = [i for i in range(len(ACTIONS))]
    self.gamma = gamma
    self.mem_size = mem_size
    self.mem_cntr = 0 # memory counter
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.replace_target = replace_target
    self.q_next = DeepQNetwork(len(ACTIONS))
    self.model = DeepQNetwork(len(ACTIONS))
    self.state_memory = np.zeros((self.mem_size, *(channels, s.ROWS, s.COLS)))
    self.new_state_memory = np.zeros((self.mem_size, *(channels, s.ROWS, s.COLS)))
    self.action_memory = np.zeros((self.mem_size, len(ACTIONS)), dtype=np.int8)
    self.reward_memory = np.zeros(self.mem_size)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8) #
    self.replace_target_cnt = 0
    


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

    # Idea: Add your own events to hand out rewards
    if ...:
        events.append(PLACEHOLDER_EVENT)

    # # state_to_features is defined in callbacks.py
    # self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # store the transitions
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state_to_features(old_game_state)
    self.new_state_memory[index] = state_to_features(new_game_state)
    action_values = np.zeros(len(ACTIONS))
    action_values[ACTIONS.index(self_action)] = 1.0
    self.action_memory[index] = action_values
    self.reward_memory[index] = reward_from_events(self, events)
    self.terminal_memory[index] = 1 if new_game_state['step'] == 399 else 0
    self.mem_cntr += 1

    # learn from the transitions
    learn(self) 




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
    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # store the transitions
    index = self.mem_cntr % self.mem_size
    self.state_memory[index] = state_to_features(last_game_state)
    self.new_state_memory[index] = state_to_features(last_game_state)
    action_values = np.zeros(len(ACTIONS))
    action_values[ACTIONS.index(last_action)] = 1.0
    self.action_memory[index] = action_values
    self.reward_memory[index] = reward_from_events(self, events)
    self.terminal_memory[index] = 1
    self.mem_cntr += 1

    # learn from the transitions
    learn(self)

    # Store the model
    # with open("my-saved-model.pt", "wb") as file:
    #     pickle.dump(self.model, file)
    # pytorch save
    torch.save(self.model.state_dict(), MODEL_NAME)


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
        e.WAITED: -3,
        e.INVALID_ACTION: -3,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -200,
        e.GOT_KILLED: -10,
        e.CRATE_DESTROYED: 50
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum

def learn(self):
    if self.mem_cntr < self.batch_size:
        return
    
    if self.mem_cntr % self.replace_target == 0:
        update_graph(self)
    
    max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
    batch = np.random.choice(max_mem, self.batch_size) # experience replay

    state_batch = self.state_memory[batch]
    state_batch = torch.from_numpy(state_batch).float()
    action_batch = self.action_memory[batch]
    action_values = np.array(self.action_space, dtype=np.int8)
    action_indices = np.dot(action_batch, action_values)
    reward_batch = self.reward_memory[batch]
    new_state_batch = self.new_state_memory[batch]
    new_state_batch = torch.from_numpy(new_state_batch).float()
    terminal_batch = self.terminal_memory[batch]


    q_eval = self.model.forward(state_batch)
    q_next = self.q_next.forward(new_state_batch)
    q_target = q_eval.clone().detach()
    batch_index = np.arange(self.batch_size, dtype=np.int32)
    # print(q_next.max(dim=1)[0])
    q_target = q_target.double()
    q_target[batch_index, action_indices] = torch.from_numpy(reward_batch + \
                                            self.gamma*q_next.max(dim=1)[0].detach().numpy()*terminal_batch)
    q_target = q_target.float()
    # train the model with pytorch
    self.model.optimizer.zero_grad()
    loss = self.model.loss(torch.Tensor(q_target), torch.Tensor(q_eval))
    loss.backward()
    self.model.optimizer.step()

    self.epsilon = self.epsilon * (1-1e-4) if self.epsilon > \
                    0.1 else 0.1
    
    # record the loss, epsilon
    self.logger.debug(f'Loss: {loss}, Epsilon: {self.epsilon}')

def save_models(self):
    self.model.save_model()
    self.q_next.save_model()

def load_models(self):
    self.model.load_model()
    self.q_next.load_model()

def update_graph(self):
    self.q_next.load_state_dict(self.model.state_dict())
    # print('... target network updated ...')
    self.replace_target_cnt += 1