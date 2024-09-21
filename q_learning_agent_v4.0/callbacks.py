import os
import pickle
from collections import defaultdict
import numpy as np
import heapq
import random
from collections import deque
from functools import lru_cache, wraps



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

    # Check if we're continuing from a saved state
    if os.path.isfile("my-saved-model.pt"):
        print("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.q_table = defaultdict(lambda: defaultdict(float), pickle.load(file))
    else:
        print("Setting up model from scratch.")
        self.q_table = defaultdict(lambda: defaultdict(float))


    self.steps_since_bomb = 6
    
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


def hash_game_state(game_state):
    return (
        game_state['field'].tobytes(),
        tuple((x, y, t) for (x, y), t in game_state['bombs']),
        tuple(game_state['coins']),
        game_state['self'],  # Include the entire self tuple
        tuple((n, s, b, (x, y)) for n, s, b, (x, y) in game_state['others']),
        game_state['explosion_map'].tobytes()
    )

@lru_cache(maxsize=10000)
def cached_state_to_features(hashed_state):
    field_bytes, bombs, coins, self_info, others, explosion_map_bytes = hashed_state
    
    game_state = {
        'field': np.frombuffer(field_bytes, dtype=int).reshape((17, 17)),
        'bombs': [((x, y), t) for x, y, t in bombs],
        'coins': list(coins),
        'self': self_info,  # Use the complete self tuple
        'others': [tuple(other) for other in others],
        'explosion_map': np.frombuffer(explosion_map_bytes, dtype=int).reshape((17, 17))
    }
    
    return original_state_to_features(game_state)

def state_to_features(game_state: dict) -> tuple:
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    # Use the cached version
    hashed_state = hash_game_state(game_state)
    return cached_state_to_features(hashed_state)

def original_state_to_features(game_state: dict) -> np.array:
    # This is your original state_to_features function
    # Copy the entire body of your current state_to_features function here
    
    if game_state is None:
        return None
    
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']
    others = game_state['others']
    position = game_state['self'][3]
        

    nearest_enemies = [enemy for enemy in others if manhattan_distance(position, enemy[3]) <= 5]
    tracked_enemy = max(nearest_enemies, key=lambda x: x[1]) if nearest_enemies else None

    # Add features
    features = []
    for dx, dy in [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)]:  # Center, Up, Right, Down, Left
        features.append(get_tile_value(position[0]+dx, position[1]+dy, field, bombs, explosion_map, others))

    direction, target_type = get_direction(game_state)
    features.append(tuple((direction, target_type)))

    crates = [(x, y) for x in range(field.shape[0]) for y in range(field.shape[1]) if field[x, y] == 1]
    enemy_direction = get_enemy_direction(position, others, field, coins, crates, bombs, explosion_map)
    features.append(enemy_direction)

    features.append(is_optimal_bomb_position(
        position, field, others, bombs, tracked_enemy, 
        target_type, enemy_direction, game_state['self'][2]
    ))


    return tuple(features)



def get_tile_value(x, y, field, bombs, explosions, others):
        if any(o[3] == (x, y) for o in others):
            return -2  # Another player is present (prioritized over bombs)
        if explosions[x, y] > 0:
            return 0  # Explosion is currently happening
        if (x, y) in [b[0] for b in bombs]:
            return -2  # Bomb is present
        if field[x, y] != 0:
            return -2  # Stone wall or crate

        # Check for future explosions
        for (bx, by), timer in bombs:
            if bx == x:  # Same column
                blocked = any(field[x, y_] == -1 for y_ in range(min(y, by) + 1, max(y, by)))
                if not blocked and abs(by - y) <= 3:
                    if timer == 0:
                        return 0  # Imminent explosion
            elif by == y:  # Same row
                blocked = any(field[x_, y] == -1 for x_ in range(min(x, bx) + 1, max(x, bx)))
                if not blocked and abs(bx - x) <= 3:
                    if timer == 0:
                        return 0  # Imminent explosion
        
        return -1  # Free tile, no danger

def is_optimal_bomb_position(position, field, others, bombs, tracked_enemy, get_direction_result, get_enemy_direction_result, can_place_bomb):
        x, y = position
        
        # Check if we can place a bomb
        if not can_place_bomb:
            return 0, None
        
        # If we're moving towards a coin or chasing an enemy, don't place bombs
        if get_direction_result == 'coin':
            return 0, None
        
        # Check if tracked enemy is trapped and within bomb range
        if tracked_enemy:
            enemy_x, enemy_y = tracked_enemy[3]
            if manhattan_distance(position, (enemy_x, enemy_y)) <= 3:  # Within bomb range
                # Check for vertical trap
                vertical_blocked = (
                    is_blocked(field, enemy_x - 1, enemy_y, others, bombs, position) and
                    is_blocked(field, enemy_x + 1, enemy_y, others, bombs, position)
                )
                # Check for horizontal trap
                horizontal_blocked = (
                    is_blocked(field, enemy_x, enemy_y - 1, others, bombs, position) and
                    is_blocked(field, enemy_x, enemy_y + 1, others, bombs, position)
                )
                
                if vertical_blocked or horizontal_blocked:
                    free_tiles = bfs_free_tiles(field, enemy_x, enemy_y, others, bombs, position)
                    if free_tiles < 3:
                        return 1, 'kill_bomb'
                    
        # If we're moving towards a crate, only place bomb when adjacent
        if get_direction_result == 'reached_crate':
            return 1, 'crate_bomb'
        
        # If chasing enemies, places bomb when adjacent (will only happen at the endgame)
        if get_direction_result == 'chasing_enemy' or get_enemy_direction_result != 0:
            # Check adjacent tiles for enemies
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]:
                    if any(enemy[3] == (nx, ny) for enemy in others):  # Enemy
                        return 1, 'enemy_bomb'
        
        return 0, None


def get_enemy_direction(position, others, field, coins, crates, bombs, explosion_map):
    if not others or (not coins and not crates):
        return 0

    # Check if we're adjacent to a safe tile
    adjacent_safe_tile = any(
        is_valid_tile(field, position[0] + dx, position[1] + dy) and
        not is_in_danger((position[0] + dx, position[1] + dy), bombs, field)
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]
    )

    nearest_enemies = []
    for enemy in others:
        if adjacent_safe_tile:
            path = a_star(field, position, enemy[3], [], others, explosion_map)
            if path and len(path) <= 7:  # Path length of 6 or less (5 steps away + current position)
                nearest_enemies.append((enemy, path))
        else:
            path = a_star(field, position, enemy[3], bombs, others, explosion_map)
            if path and len(path) <= 7:  # Path length of 6 or less (5 steps away + current position)
                nearest_enemies.append((enemy, path))
    
    if not nearest_enemies:
        return 0
    
    # Choose the highest scoring enemy among the nearest ones
    highest_scoring_enemy, path = max(nearest_enemies, key=lambda x: x[0][1])
    
    if len(path) > 1:
        return get_direction_from_path(position, path[1])
    return 0

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
            prioritized_safe_tiles = find_prioritized_safe_tiles(field, bombs, explosion_map, others, position, coins) 
            if prioritized_safe_tiles:              
                path = a_star_to_safety(field, position, prioritized_safe_tiles, bombs, others, explosion_map)
                if path and len(path) > 1:
                    return get_direction_from_path(position, path[1]), 'running_from_danger'
        
        # If no path to safety is found, fall back to the previous method
            return get_safest_direction(field, position, bombs, others, explosion_map), 'running_from_danger'
        
        # If we've placed a bomb and we're safe, wait for explosion
        if not can_place_bomb and not is_in_danger(position, bombs, field):
            # Check if there are collectable coins
            for coin in coins:
                if is_path_clear(field, position, coin, bombs, others, explosion_map):
                    path = a_star(field, position, coin, bombs, others, explosion_map)
                    if path and len(path) > 1:
                        return get_direction_from_path(position, path[1]), 'coin'
            # If no collectable coins, wait
            return 0, 'wait'  # 0 represents "WAIT"
        
        # Priority 1: Coins with unobstructed path
        reachable_coin = find_reachable_coin(position, coins, others, field, bombs)
        if reachable_coin:
            path = a_star_to_coin(field, position, reachable_coin, bombs, others)
            if path and len(path) > 1:
                return get_direction_from_path(position, path[1]), 'coin'
        

        # Priority 2: Crates
        MAX_BOMB_DISTANCE = 6  # Maximum distance to consider for bomb placement
        crate_positions = [(x, y) for x in range(field.shape[0]) for y in range(field.shape[1]) if field[x, y] == 1]
        if crate_positions:
            # Find all valid positions to place bombs (empty tiles) within MAX_BOMB_DISTANCE
            bomb_positions = [(x, y) for x in range(field.shape[0]) for y in range(field.shape[1]) 
                            if field[x, y] == 0 and 
                                not any(b[0] == (x,y) for b in bombs) and
                                manhattan_distance(position, (x,y)) <= MAX_BOMB_DISTANCE and 
                                not ((x == 1 or x == field.shape[0] - 2) and (y == 1 or y == field.shape[1] - 2))]
                                
            # Score each position based on destroyable crates and distance
            scored_positions = []
            for bomb_pos in bomb_positions:
                field_tuple = tuple(tuple(row) for row in field)
                crates_destroyed = count_destroyable_crates(field_tuple, bomb_pos)
                distance = manhattan_distance(position, bomb_pos)
                score = crates_destroyed * 5 - distance * 2  # Prioritize crate destruction over distance
                scored_positions.append((score, bomb_pos))
            
            # Sort positions by score
            scored_positions.sort(reverse=True)
            
            # Try to find a path to the best positions
            for _, bomb_pos in scored_positions[:5]:  # Check the 5 best positions
                if manhattan_distance(position, bomb_pos) == 0:
                    # We're already at the optimal position, point to the nearest crate
                    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                        nx, ny = bomb_pos[0] + dx, bomb_pos[1] + dy
                        if 0 <= nx < field.shape[0] and 0 <= ny < field.shape[1] and field[nx, ny] == 1:
                            return get_direction_from_path(position, (nx, ny)), 'reached_crate'
                else:
                    path = a_star(field, position, bomb_pos, bombs, others, explosion_map)
                    if path and len(path) > 1:
                        return get_direction_from_path(position, path[1]), 'crate'
        
        # Priority 3: Highest scoring enemy or maximize distance
        if others:
            highest_scoring_enemy = max(others, key=lambda x: x[1])
            we_have_highest_score = own_score >= highest_scoring_enemy[1]

            if not we_have_highest_score:
                # Chase the highest scoring enemy
                path = a_star(field, position, highest_scoring_enemy[3], bombs, others, explosion_map)
                if path and len(path) > 1:
                    return get_direction_from_path(position, path[1]), 'chasing_enemy'
            else:
                # We have the highest score, so maximize distance from all enemies
                directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
                valid_directions = [d for d in directions if is_valid_tile(field, position[0]+d[0], position[1]+d[1])]
                
                if valid_directions:
                    best_direction = max(valid_directions,
                                        key=lambda d: sum(manhattan_distance((position[0]+d[0], position[1]+d[1]), enemy[3]) for enemy in others))
                    return get_direction_from_path(position, (position[0]+best_direction[0], position[1]+best_direction[1])), 'dodging_enemy'
        
        # If we reach here, there are no valid moves, so we wait
        return 0, 'wait'  # WAIT



@lru_cache(maxsize=2000)
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

@lru_cache(maxsize=1000)
def get_adjacent_tiles(pos):
    x, y = pos
    return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]


def is_valid_tile(field, x, y):
    return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] == 0


def is_in_danger(position, bombs, field):
    x, y = position
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
    return False

def find_prioritized_safe_tiles(field, bombs, explosion_map, others, agent_pos, coins, max_coin_distance=8):
    safe_tiles = []
    safe_tiles_with_coins = []
    
    for x in range(field.shape[0]):
        for y in range(field.shape[1]):
            if field[x, y] == 0 and not any(o[3] == (x, y) for o in others):
                # Only consider explosion_map for adjacent tiles
                if manhattan_distance(agent_pos, (x, y)) <= 1 and explosion_map[x, y] > 0:
                    continue
                if not is_in_danger((x, y), bombs, field):
                    distance = manhattan_distance(agent_pos, (x, y))
                    if (x, y) in coins and distance <= max_coin_distance:
                        safe_tiles_with_coins.append(((x, y), distance))
                    else:
                        safe_tiles.append(((x, y), distance))
    
    # Sort safe tiles with coins by distance
    safe_tiles_with_coins.sort(key=lambda x: x[1])
    # Sort regular safe tiles by distance
    safe_tiles.sort(key=lambda x: x[1])
    
    # Combine the lists, prioritizing safe tiles with coins
    prioritized_safe_tiles = [tile for tile, _ in safe_tiles_with_coins + safe_tiles]
    
    return prioritized_safe_tiles


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
    

def a_star_to_safety(field, start, prioritized_safe_tiles, bombs, others, explosion_map):
    def heuristic(a, b):
        return manhattan_distance(a, b)
    
    def is_valid(x, y):
        return 0 <= x < field.shape[0] and 0 <= y < field.shape[1] and field[x, y] != -1

    def get_neighbors_to_safety(pos):
        x, y = pos
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            next_x, next_y = x + dx, y + dy
            if is_valid(next_x, next_y) and field[next_x, next_y] != 1 and \
               not any(o[3] == (next_x, next_y) for o in others):
                neighbors.append((next_x, next_y))
        return neighbors

    closed_set = set()
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, prioritized_safe_tiles[0])}  # Use the first (highest priority) safe tile

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current in prioritized_safe_tiles:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        closed_set.add(current)

        for neighbor in get_neighbors_to_safety(current):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in [i[1] for i in open_set]:
                heapq.heappush(open_set, (f_score.get(neighbor, float('inf')), neighbor))
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, prioritized_safe_tiles[0])

    return None  # No path found

def count_free_tiles(field, x, y, directions, others, bombs, self_position):
    count = 0
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        direction_count = 0
        while not is_blocked(field, nx, ny, others, bombs, self_position):
            direction_count += 1
            if direction_count + count >= 3:
                return 3  # We've found at least 3 free tiles, no need to continue
            nx, ny = nx + dx, ny + dy
        count += direction_count
    return count

def is_blocked(field, x, y, others, bombs, self_position):
    if not (0 <= x < field.shape[0] and 0 <= y < field.shape[1]):
        return True
    return (field[x, y] != 0 or
            any(enemy[3] == (x, y) for enemy in others) or
            any(bomb[0] == (x, y) for bomb in bombs) or
            (x, y) == self_position)

def bfs_free_tiles(field, start_x, start_y, others, bombs, self_position):
    queue = deque([(start_x, start_y)])
    visited = set([(start_x, start_y)])
    free_tiles = 0  # Initialize to 0 instead of 1
    
    while queue and free_tiles < 3:
        x, y = queue.popleft()
        
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in visited and not is_blocked(field, nx, ny, others, bombs, self_position):
                queue.append((nx, ny))
                visited.add((nx, ny))
                free_tiles += 1  # Increment free_tiles here
                if free_tiles >= 3:
                    break
    return free_tiles


@lru_cache(maxsize=5000)
def count_destroyable_crates(field_tuple, position):
    field = np.array(field_tuple)  # Convert back to numpy array
    x, y = position
    count = 0
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        for i in range(1, 4):  # Check up to 3 tiles in each direction
            nx, ny = x + i*dx, y + i*dy
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                break
            if field[nx, ny] == -1:  # Stop at walls
                break
            if field[nx, ny] == 1:  # Count crates
                count += 1
    return count

def a_star_to_coin(field, start, goal, bombs, others):
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

def find_reachable_coin(position, coins, others, field, bombs):
    # Sort coins by Manhattan distance
    sorted_coins = sorted(coins, key=lambda c: manhattan_distance(position, c))
    
    for coin in sorted_coins:
        our_path = a_star_to_coin(field, position, coin, bombs, others)
        if not our_path:
            continue  # If we can't reach the coin, move to the next one
        
        our_distance = len(our_path) - 1
        
        enemy_can_reach_first = False
        for enemy in others:
            enemy_path = a_star_to_coin(field, enemy[3], coin, bombs, others)
            enemy_distance = len(enemy_path) - 1 if enemy_path else float('inf')
            if enemy_distance < our_distance:
                enemy_can_reach_first = True
                break
        
        if not enemy_can_reach_first:
            print(f"chosen coin: {coin}")
            return coin
    
    return None