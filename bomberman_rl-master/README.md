# bomberman_rl
Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.


## Test

Test code for task 1-4 in Main.py, if you want to show, just delete "--no-gui" in the corresponding command line:
- Task 1: `python main.py play --no-gui --agents our_agent1 --train 1 --scenario coin-heaven`
- Task 2: `python main.py play --no-gui --agents our_agent1 --train 1 --scenario classic`
- Task 3: `python main.py play --no-gui --agents our_agent1 peaceful_agent coin_collector_agent --train 1 --scenario classic`
- Task 4: `python main.py play --no-gui --agents our_agent1 rule_based_agent --train 1--scenario classic`

## Task 1

Use RL to improve the performance of our agent.

State_to_features: Output is 2D array (x, y, c) where x, y represent the rows and cols and c represents the channel.

```python
# Values
FREE = 0
WALL = -1
CRATE = 1
COIN = 2
# Agent
SELF = 3
OTHER = 4
# Bomb: actually we could also use the timer of the bomb as a feature
BOMB0 = 5 
BOMB1 = 6
BOMB2 = 7
BOMB3 = 8 # almost exploding
```

Then we can convert the game_state to the S (Maybe we could add neural network here to extract the features of the state).

