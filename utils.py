import numpy as np

def get_action(q_values, epsilon):
    if np.random.rand() < epsilon:
        # Exploration: Choose a random action
        action = np.random.randint(len(q_values))
    else:
        # Exploitation: Choose the best known action
        action = np.argmax(q_values)
    return action
