import torch 
import matplotlib as mpl
mpl.use('Agg')
import events as e
from .ManagerFeatures import state_to_features

    
ACTIONS_IDX = {'LEFT':0, 'RIGHT':1, 'UP':2, 'DOWN':3, 'WAIT':4, 'BOMB':5}


def add_experience(self, old_game_state, self_action, new_game_state, events):
    '''
    fills the experience buffer

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: dict contoining the old game state
    :param self_action: action executed by the agent
    :param new_game_state: dict contoining the new game state
    :param events: events that occured 
    :param n: n-step Q-learning, default 5
    '''
    old_features = state_to_features(self, old_game_state)
    if old_features is not None:
        if new_game_state is None:
            new_features = old_features
        else:
            new_features = state_to_features(self, new_game_state)
        reward = reward_from_events(self, events)
        reward += count_destroyed_crates(events,old_game_state)

        # encode the actions into hot-one, needed in train_network()
        action_idx = ACTIONS_IDX[self_action]
        action = torch.zeros(6)
        action[action_idx] = 1


        self.experience_buffer.append((old_features, action, reward, new_features))

        number_of_elements_in_buffer = len(self.experience_buffer)

        if number_of_elements_in_buffer > self.network.buffer_size:
            self.experience_buffer.pop(0)


def reward_from_events(self, events) -> int:
    '''
    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param events: pre defined events (events.py) that occured in game step

    return: reward based on events in (events.py)
    '''
    game_rewards = {
        e.COIN_COLLECTED: 100,
        e.KILLED_OPPONENT: 500,
        e.MOVED_RIGHT: -1,
        e.MOVED_LEFT: -1,
        e.MOVED_UP: -1,
        e.MOVED_DOWN: -1,
        e.WAITED: -1,
        e.INVALID_ACTION: -10,
        e.BOMB_DROPPED: -1,
        e.KILLED_SELF: -250,
        e.GOT_KILLED: -500,
    }

    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


    
def count_destroyed_crates(events,new_game_state):
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
    
    return destroyed_count * 33


    
 

