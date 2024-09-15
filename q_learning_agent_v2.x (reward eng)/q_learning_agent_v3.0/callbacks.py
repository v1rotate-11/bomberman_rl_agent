import os
import pickle
import random
from collections import defaultdict
import numpy as np
import heapq

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
    
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    position = game_state['self'][3]
    can_place_bomb = game_state['self'][2]
    own_score = game_state['self'][1]
    
    def get_direction(game_state):
        field = game_state['field']
        coins = game_state['coins']
        bombs = game_state['bombs']
        explosion_map = game_state['explosion_map']
        others = game_state['others']
        position = game_state['self'][3]
        can_place_bomb = game_state['self'][2]
        own_score = game_state['self'][1]
        
        # Check if agent is in danger from bombs
        if is_in_danger(position, bombs, field):
            safe_directions = get_safe_directions(field, position, bombs, others, explosion_map)
            if safe_directions:
                return safe_directions[0], "safety"
            else:
                return get_safest_direction(field, position, bombs, others, explosion_map), "safety"
        
        # If we've placed a bomb and we're safe, wait for explosion
        if not can_place_bomb and not is_in_danger(position, bombs, field):
            # Check if there are collectable coins
            for coin in coins:
                if is_path_clear(field, position, coin, bombs, others, explosion_map):
                    path = a_star(field, position, coin, bombs, others, explosion_map)
                    if path and len(path) > 1:
                        return get_direction_from_path(position, path[1]), "coin"
            # If no collectable coins, wait
            return 0, "wait"  # 0 represents "WAIT"
        
        # Priority 1: Coins with unobstructed path
        for coin in coins:
            path = a_star(field, position, coin, bombs, others, explosion_map)
            if path and len(path) > 1:
                return get_direction_from_path(position, path[1]), "coin"
        
        # Priority 2: Crates
        crates = [(x, y) for x in range(field.shape[0]) for y in range(field.shape[1]) if field[x, y] == 1]
        if crates:
            # Sort crates by distance
            crates.sort(key=lambda c: manhattan_distance(position, c))
            for crate in crates[:5]:  # Check the 5 closest crates
                # Check if we're adjacent to the crate
                if manhattan_distance(position, crate) == 1:
                    return get_direction_from_path(position, crate), "crate"
                else:
                    # Find a path to a tile adjacent to the crate
                    adjacent_tiles = get_adjacent_tiles(crate)
                    valid_adjacent_tiles = [tile for tile in adjacent_tiles if is_valid_tile(field, tile[0], tile[1])]
                    for tile in valid_adjacent_tiles:
                        path = a_star(field, position, tile, bombs, others, explosion_map)
                        if path and len(path) > 1:
                            return get_direction_from_path(position, path[1]), "crate"
        
        # Priority 3: Highest scoring enemy or maximize distance
        if others:
            highest_scoring_enemy = max(others, key=lambda x: x[1])
            we_have_highest_score = own_score >= highest_scoring_enemy[1]

            if not we_have_highest_score:
                # Chase the highest scoring enemy
                path = a_star(field, position, highest_scoring_enemy[3], bombs, others, explosion_map)
                if path and len(path) > 1:
                    return get_direction_from_path(position, path[1]), "enemy"
            else:
                # We have the highest score, so maximize distance from all enemies
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                valid_directions = [d for d in directions if is_valid_tile(field, position[0]+d[0], position[1]+d[1])]
                
                if valid_directions:
                    best_direction = max(valid_directions,
                                        key=lambda d: sum(manhattan_distance((position[0]+d[0], position[1]+d[1]), enemy[3]) for enemy in others))
                    return get_direction_from_path(position, (position[0]+best_direction[0], position[1]+best_direction[1])), "avoid_enemy"
        
        # If we reach here, there are no valid moves, so we wait
        return 0, "wait"  # WAIT
    
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

    
    # Add features
    features = []
    for dx, dy in [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]:  # Center, Up, Right, Down, Left
        features.append(get_tile_value(position[0]+dx, position[1]+dy, field, bombs, explosion_map, others))
    
    features.append(game_state['self'][2])  # Can place bomb
    features.append(get_direction(game_state))

    def is_optimal_bomb_position(position, field, others):
        x, y = position
        # Check adjacent tiles for crates or enemies
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                if field[nx, ny] == 1:  # Crate
                    return 1
                if any(enemy[3] == (nx, ny) for enemy in others):  # Enemy
                    return 1
        return 0
    
    if others:
        nearest_enemy_dist = min(manhattan_distance(position, enemy[3]) for enemy in others)
        if nearest_enemy_dist > 10:
            features.append(11)  # Represent all distances > 10 with 11
        else:
            features.append(nearest_enemy_dist)
    else:
        features.append(11)  # No enemies on the field
    
    features.append(is_optimal_bomb_position(position, field, others))
    return tuple(features)


def manhattan_distance(pos1, pos2):
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def get_direction_from_path(start, next_pos):
    dx, dy = next_pos[0] - start[0], next_pos[1] - start[1]
    if dx == 1: return 2  # RIGHT
    if dx == -1: return 4  # LEFT
    if dy == -1: return 1  # UP
    if dy == 1: return 3  # DOWN
    return 0  # Same position

def is_path_clear(field, start, end, bombs, others, explosion_map):
    """Check if there's a clear path from start to end without obstacles or dangers."""
    path = a_star(field, start, end, bombs, others, explosion_map)
    return path is not None and len(path) > 1

def get_closest_points(start, points):
    min_dist = float('inf')
    closest = []
    for point in points:
        dist = manhattan_distance(start, point)
        if dist < min_dist:
            min_dist = dist
            closest = [point]
        elif dist == min_dist:
            closest.append(point)
    return closest

def get_adjacent_tiles(pos):
    x, y = pos
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

def is_valid_tile(field, x, y):
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0

def is_in_danger(position, bombs, field):
    for bomb_pos, bomb_timer in bombs:
        if (position[0] == bomb_pos[0] and abs(position[1] - bomb_pos[1]) <= 3) or \
           (position[1] == bomb_pos[1] and abs(position[0] - bomb_pos[0]) <= 3):
            # Check if there's a wall blocking the explosion
            if position[0] == bomb_pos[0]:
                if not any(field[position[0], y] == -1 for y in range(min(position[1], bomb_pos[1]), max(position[1], bomb_pos[1]))):
                    return True
            else:
                if not any(field[x, position[1]] == -1 for x in range(min(position[0], bomb_pos[0]), max(position[0], bomb_pos[0]))):
                    return True


def get_safe_directions(field, position, bombs, others, explosion_map):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    safe_directions = []
    for d in directions:
        new_pos = (position[0] + d[0], position[1] + d[1])
        if is_valid_tile(field, new_pos[0], new_pos[1]) and \
           not is_in_danger(new_pos, bombs, field) and \
           not any(o[3] == new_pos for o in others) and \
           explosion_map[new_pos[0], new_pos[1]] == 0:
            safe_directions.append(get_direction_from_path(position, new_pos))
    return safe_directions

def get_safest_direction(field, position, bombs, others, explosion_map):
    directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    safest_direction = 0
    max_safety = -float('inf')
    
    for d in directions:
        new_pos = (position[0] + d[0], position[1] + d[1])
        if is_valid_tile(field, new_pos[0], new_pos[1]) and \
           not any(o[3] == new_pos for o in others) and \
           explosion_map[new_pos[0], new_pos[1]] == 0:
            safety_score = calculate_safety_score(new_pos, bombs, field)
            if safety_score > max_safety:
                max_safety = safety_score
                safest_direction = get_direction_from_path(position, new_pos)
    
    return safest_direction

def calculate_safety_score(position, bombs, field):
    safety_score = 0
    for bomb_pos, bomb_timer in bombs:
        distance = manhattan_distance(position, bomb_pos)
        if distance <= 3:
            if (position[0] == bomb_pos[0] or position[1] == bomb_pos[1]) and \
               not is_blocked_by_wall(position, bomb_pos, field):
                safety_score -= (4 - distance) * (5 - bomb_timer)
        else:
            safety_score += distance
    return safety_score

def is_blocked_by_wall(pos1, pos2, field):
    if pos1[0] == pos2[0]:  # Same column
        for y in range(min(pos1[1], pos2[1]), max(pos1[1], pos2[1])):
            if field[pos1[0], y] == -1:
                return True
    else:  # Same row
        for x in range(min(pos1[0], pos2[0]), max(pos1[0], pos2[0])):
            if field[x, pos1[1]] == -1:
                return True
    return False

def a_star(field, start, goal, bombs, others, explosion_map):
    def heuristic(a, b):
        return manhattan_distance(a, b)
    
    def is_valid(x, y):
        return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] != -1

    def get_neighbors(pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if is_valid(next_x, next_y) and field[next_x, next_y] != 1 and \
               not is_in_danger((next_x, next_y), bombs, field) and \
               explosion_map[next_x, next_y] == 0 and \
               ((next_x, next_y) == goal or not any(o[3] == (next_x, next_y) for o in others)):
                neighbors.append((next_x, next_y))
        return neighbors

    closed_set = set()
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in [i[1] for i in open_set]:
                heapq.heappush(open_set, (f_score.get(neighbor, float('inf')), neighbor))
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)

    return None  # No path found
