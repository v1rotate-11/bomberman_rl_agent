import os
import pickle
import random

import numpy as np
TRANSITION_HISTORY_SIZE = 50000

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


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
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.q_table = {}
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    # todo Exploration vs exploitation
    features = tuple(state_to_features(game_state))
    
    # Epsilon-greedy action selection
    epsilon = max(0.01, min(1, 1.0 - len(self.transitions) / TRANSITION_HISTORY_SIZE))
    if self.train and random.random() < epsilon:
        self.logger.debug("Choosing action randomly")
        return np.random.choice(ACTIONS)
    
    self.logger.debug("Choosing action greedily")
    # Check if the state is completely unknown
    if features not in self.q_table or all(self.q_table[features].get(a, 0) == 0 for a in ACTIONS):
        self.logger.debug("Unknown state, choosing random action")
        return np.random.choice(ACTIONS)
    return max(ACTIONS, key=lambda a: self.q_table.get(features, {}).get(a, 0))

def state_to_features(game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    # Extract relevant information from game_state
    field = game_state['field']
    position = game_state['self'][3]
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    
    # Create features
    features = []
    
    # Agent's position
    features.extend(position)
    
    # Nearby tiles (up, right, down, left)
    for dx, dy in [(0,-1), (1,0), (0,1), (-1,0)]:
        features.append(field[position[0]+dx, position[1]+dy])
    
    # Can agent drop a bomb?
    features.append(game_state['self'][2])
    
    # Distance and direction to nearest coin
    if coins:
        nearest_coin = min(coins, key=lambda c: abs(c[0]-position[0]) + abs(c[1]-position[1]))
        features.append(nearest_coin[0] - position[0])
        features.append(nearest_coin[1] - position[1])
    else:
        features.extend([-1, -1])  # No coins left
    
    # Information about bombs
    if bombs:
        nearest_bomb = min(bombs, key=lambda b: abs(b[0][0]-position[0]) + abs(b[0][1]-position[1]))
        features.append(nearest_bomb[0][0] - position[0])
        features.append(nearest_bomb[0][1] - position[1])
        features.append(nearest_bomb[1])  # Timer
    else:
        features.extend([-1, -1, -1])
    
    # Danger level (from explosion map)
    danger_level = 0
    for dx, dy in [(0,0), (0,1), (1,0), (0,-1), (-1,0)]:
        x, y = position[0]+dx, position[1]+dy
        if field[x, y] != -1:  # -1 represents walls in this game
            danger_level = max(danger_level, explosion_map[x, y])  
    features.append(danger_level)
      
    # Information about other agents
    if others:
        nearest_other = min(others, key=lambda o: abs(o[3][0]-position[0]) + abs(o[3][1]-position[1]))
        features.append(nearest_other[3][0] - position[0])
        features.append(nearest_other[3][1] - position[1])
    else:
        features.extend([-1, -1])
    
    return np.array(features)
