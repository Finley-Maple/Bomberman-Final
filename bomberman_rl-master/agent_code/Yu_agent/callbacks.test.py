import numpy as np
from callbacks import state_to_features

# Define a sample game state
game_state = {
    'field': np.array([
        [ 1,  0,  0,  0,  0],
        [ 0,  1,  1,  1,  0],
        [ 0,  1, -1,  1,  0],
        [ 0,  1,  1,  1,  0],
        [ 0,  0,  0,  0,  0]
    ]),
    'self': (0, 0, 3, (1, 1)),
    'others': [],
    'coins': np.array([[3, 1]]),
    'bombs': []
}

# Call the state_to_features function
features = state_to_features(game_state)

print(features.shape)

# Print the resulting feature vector
print(features)

