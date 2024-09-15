import os
import pickle
import random
from collections import defaultdict
import numpy as np

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

    self.current_epsilon = 1.0

    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.q_table = defaultdict(lambda: defaultdict(float))
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
    
    if self.train and random.random() < self.current_epsilon:
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
    
    position = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    
    def get_direction_to_nearest_coin():
        if not coins:
            return 0  # No coins available

        # Find nearest coins
        distances = [manhattan_distance(position, coin) for coin in coins]
        min_distance = min(distances)
        nearest_coins = [coin for coin, dist in zip(coins, distances) if dist == min_distance]

        if len(nearest_coins) == 1:
            return get_direction(position, nearest_coins[0])
        
        # Handle multiple equidistant coins
        safest_coin = nearest_coins[0]
        max_bomb_distance = -1
        for coin in nearest_coins:
            if not bombs:
                bomb_distance = float('inf')
            else:
                bomb_distance = min(manhattan_distance(coin, bomb[0]) for bomb in bombs)
            if bomb_distance > max_bomb_distance:
                safest_coin = coin
                max_bomb_distance = bomb_distance
        
        return get_direction(position, safest_coin)

    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_direction(from_pos, to_pos):
        dx, dy = to_pos[0] - from_pos[0], to_pos[1] - from_pos[1]
        if dx == 0 and dy == 0:
            return 0  # Same position
        if abs(dx) > abs(dy):
            return 2 if dx > 0 else 4  # Right or Left
        return 1 if dy < 0 else 3  # Up or Down
    
    def get_tile_value(x, y, field, bombs, explosions, others):
        if any(o[3] == (x, y) for o in others):
            return -4  # Another player is present (prioritized over bombs)
        if explosions[x, y] > 0:
            return -6  # Explosion is currently happening
        if (x, y) in [b[0] for b in bombs]:
            return -5  # Bomb is present
        if field[x, y] == -1:
            return -3  # Stone wall
        if field[x, y] == 1:
            return -2  # Crate

        # Check for future explosions
        min_time = float('inf')
        for (bx, by), timer in bombs:
            if bx == x:  # Same column
                blocked = any(field[x, y_] == -1 for y_ in range(min(y, by) + 1, max(y, by)))
                if not blocked and abs(by - y) <= 3:
                    min_time = min(min_time, timer)
            elif by == y:  # Same row
                blocked = any(field[x_, y] == -1 for x_ in range(min(x, bx) + 1, max(x, bx)))
                if not blocked and abs(bx - x) <= 3:
                    min_time = min(min_time, timer)
        
        if min_time == float('inf'):
            return -1  # Free tile, no danger
        return min(3, min_time)  # Cap at 3 for future explosions

    features = []
    for dx, dy in [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]:  # Center, Up, Right, Down, Left
        features.append(get_tile_value(position[0]+dx, position[1]+dy, field, bombs, explosion_map, others))
    
    features.append(game_state['self'][2])  # Can place bomb
    features.append(get_direction_to_nearest_coin(game_state))
    
    if others:
        nearest_other = min(others, key=lambda o: manhattan_distance(position, o[3]))
        features.extend([nearest_other[3][0] - position[0], nearest_other[3][1] - position[1]])
    else:
        features.extend([-1, -1])
    
    return tuple(features)
