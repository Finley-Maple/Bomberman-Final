from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
from RL_brain import DeepQNetwork
import settings as s
import numpy as np

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
alpha = 0.0001 # learning rate
replace_target = 100 # replace target network after ... steps
q_next_dir = 'q_next'
q_eval_dir = 'q_eval'


# Action space without the bomb action
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

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
    self.mem_cntr = 0
    self.epsilon = epsilon
    self.batch_size = batch_size
    self.replace_target = replace_target
    self.q_next = DeepQNetwork(alpha, len(ACTIONS), input_dims=(s.ROWS, s.COLS), 
                                name='q_next', fct_dims=512)
    self.model = DeepQNetwork(alpha, len(ACTIONS), input_dims=(s.ROWS, s.COLS), 
                                name='q_eval', fct_dims=512)
    self.state_memory = np.zeros((self.mem_size, *(s.ROWS, s.COLS)))
    self.new_state_memory = np.zeros((self.mem_size, *(s.ROWS, s.COLS)))
    self.action_memory = np.zeros((self.mem_size, len(ACTIONS)), dtype=np.int8)
    self.reward_memory = np.zeros(self.mem_size)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.int8)
    # self.learn_step_counter = 0
    # self.q_next_dir = q_next_dir
    # self.q_eval_dir = q_eval_dir
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

    # state_to_features is defined in callbacks.py
    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))

    # learn from the transitions
    self.learn() 

def learn(self):
    if self.mem_cntr % self.replace_target == 0:
        self.update_graph()
    
    max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
    batch = np.random.choice(max_mem, self.batch_size) # experience replay

    state_batch = self.state_memory[batch]
    action_batch = self.action_memory[batch]
    action_values = np.array(self.action_space, dtype=np.int8)
    action_indices = np.dot(action_batch, action_values)
    reward_batch = self.reward_memory[batch]
    new_state_batch = self.new_state_memory[batch]
    terminal_batch = self.terminal_memory[batch]

    q_eval = self.model.sess.run(self.model.Q_values, 
                                    feed_dict={self.model.input:new_state_batch})
    q_next = self.q_next.sess.run(self.q_next.Q_values, 
                                    feed_dict={self.q_next.input:new_state_batch})
    q_target = q_eval.copy()
    batch_index = np.arange(self.batch_size, dtype=np.int32)
    q_target[batch_index, action_indices] = reward_batch + \
                                            self.gamma*np.max(q_next, axis=1)*terminal_batch

    _ = self.model.sess.run(self.model.train_op, 
                                feed_dict={self.model.input:state_batch, 
                                            self.model.actions:q_target})
    self.epsilon = self.epsilon * (1-1e-5) if self.epsilon > \
                    0.01 else 0.01
    self.learn_step_counter += 1


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

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.model, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.KILLED_SELF: -10, # idea: killing yourself is bad
        PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
