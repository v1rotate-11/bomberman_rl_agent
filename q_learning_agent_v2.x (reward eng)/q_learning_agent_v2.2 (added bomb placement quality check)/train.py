from collections import namedtuple, deque
from typing import List, Tuple
import random
from .callbacks import ACTIONS
import pickle
from typing import List

from ..Tracker import MultiSessionPerformanceTracker
# from ..PerformanceTracker import SimplePerformanceTracker
# from ..AdvancedPerformanceTracker import RefinedPerformanceTracker
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
N_TRAINING_EPSD = 800000
BATCH_SIZE = 32  # mini-batch size for training
TRAIN_FREQ = 4

# Events
MOVED_TOWARDS_COIN = "MOVED_TOWARDS_COIN"
MOVED_AWAY_FROM_COIN = "MOVED_AWAY_FROM_COIN"
FOLLOWED_COIN_DIRECTION = "FOLLOWED_COIN_DIRECTION"
MOVED_TOWARDS_ENEMY = "MOVED_TOWARDS_ENEMY"
MOVED_AWAY_FROM_ENEMY = "MOVED_AWAY_FROM_ENEMY"
FOLLOWED_ENEMY_DIRECTION = "FOLLOWED_ENEMY_DIRECTION"
BOMB_PLACED_NEAR_ENEMY = "BOMB_PLACED_NEAR_ENEMY"
BOMB_PLACED_NEAR_CRATE = "BOMB_PLACED_NEAR_CRATE"
BOMB_PLACED_INEFFECTIVELY = "BOMB_PLACED_INEFFECTIVELY"
BOMB_PLACED_UNSAFELY = "BOMB_PLACED_UNSAFELY" 
BOMB_PLACED_RISKILY = "BOMB_PLACED_RISKILY" 


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.enemy_decay_rate = 0.9999  # decay rate for enemy transition recording probability
    self.current_enemy_prob = RECORD_ENEMY_TRANSITIONS
    self.episode_reward = 0
    self.survival_time = 0
    self.coins_collected = 0
    self.tracker = MultiSessionPerformanceTracker()
    self.prob_record = RECORD_ENEMY_TRANSITIONS
    self.random_value = 0
    self.current_step = 0
    self.steps_since_train = 0
    
    def train_step():
        if len(self.transitions) < BATCH_SIZE:
            return

        # Sample a mini-batch
        mini_batch = random.sample(self.transitions, BATCH_SIZE)

        # Compute the loss and perform an optimization step
        for state, action, next_state, reward in mini_batch:
            current_q = self.q_table[state][action]

            if next_state is not None:
                # Non-terminal state
                max_next_q = max(self.q_table[next_state].values(), default=0)
                target_q = reward + 0.9 * max_next_q  # 0.9 is the discount factor
            else:
                # Terminal state
                target_q = reward

            # Update Q-value
            self.q_table[state][action] += 0.1 * (target_q - current_q)  # 0.1 is the learning rate

    self.train_step = train_step



def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def check_escape_routes(game_state: dict, bomb_position: Tuple[int, int]) -> str:
    def is_valid_tile(x: int, y: int) -> bool:
        return 0 <= x < game_state['field'].shape[0] and 0 <= y < game_state['field'].shape[1]

    def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    agent_pos = game_state['self'][3]
    escape_route_exists = False
    safe_escape_route_exists = False

    for dx, dy in directions:
        turn_found = False
        for i in range(1, 5):  # Check up to 4 tiles in each direction
            x, y = bomb_position[0] + i*dx, bomb_position[1] + i*dy
            if not is_valid_tile(x, y) or game_state['field'][x, y] != 0:
                break  # Hit a wall or crate
            
            if i == 4 or turn_found:
                # Potential escape tile found
                escape_route_exists = True
                escape_tile = (x, y)
                agent_distance = manhattan_distance(agent_pos, escape_tile)
                
                # Check if any enemy is closer to the escape tile
                enemy_closer = False
                for enemy in game_state['others']:
                    if manhattan_distance(enemy[3], escape_tile) < agent_distance:
                        enemy_closer = True
                        break
                
                if not enemy_closer:
                    safe_escape_route_exists = True
                    return "SAFE"  # Safe escape route found

            # Check for turns
            if not turn_found:
                for turn_dx, turn_dy in directions:
                    if (turn_dx, turn_dy) != (dx, dy) and (turn_dx, turn_dy) != (-dx, -dy):
                        turn_x, turn_y = x + turn_dx, y + turn_dy
                        if is_valid_tile(turn_x, turn_y) and game_state['field'][turn_x, turn_y] == 0:
                            turn_found = True
                            break

    if escape_route_exists:
        return "RISKY"  # Escape route exists, but enemy might reach it first
    else:
        return "UNSAFE"  # No escape route found


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
    if self.current_step != new_game_state['step']:
        self.current_step = new_game_state['step']
        self.random_value = random.random()
        self.steps_since_train += 1

    old_features = tuple(state_to_features(old_game_state))
    new_features = tuple(state_to_features(new_game_state))

    # Coin direction related features
    old_coin_direction = old_features[6]  # Coin direction is the 7th feature
    new_position = new_game_state['self'][3]
    old_position = old_game_state['self'][3]

    # Enemy direction related features
    old_enemy_direction = old_features[7]  # Enemy direction is the 8th feature

    # Calculate actual move once
    actual_move = (new_position[0] - old_position[0], new_position[1] - old_position[1])

    # Check coin-related events
    followed_coin_direction = False
    if old_coin_direction != 0:  # Only if there was a suggested direction
        suggested_move = {
            1: (0, -1),  # Up
            2: (1, 0),   # Right
            3: (0, 1),   # Down
            4: (-1, 0)   # Left
        }.get(old_coin_direction, (0, 0))
        
        if actual_move == suggested_move:
            events.append(FOLLOWED_COIN_DIRECTION)
            followed_coin_direction = True
    
    if not followed_coin_direction and old_game_state['coins'] and new_game_state['coins']:
        old_nearest_coin = min(old_game_state['coins'], key=lambda c: manhattan_distance(old_position, c))
        new_nearest_coin = min(new_game_state['coins'], key=lambda c: manhattan_distance(new_position, c))
        
        old_distance = manhattan_distance(old_position, old_nearest_coin)
        new_distance = manhattan_distance(new_position, new_nearest_coin)
        
        if new_distance < old_distance:
            events.append(MOVED_TOWARDS_COIN)
        elif new_distance > old_distance:
            events.append(MOVED_AWAY_FROM_COIN)

    # Check enemy-related events
    followed_enemy_direction = False
    if old_enemy_direction != 0:  # Only if there was a suggested direction
        suggested_move = {
            1: (0, -1),  # Up
            2: (1, 0),   # Right
            3: (0, 1),   # Down
            4: (-1, 0)   # Left
        }.get(old_enemy_direction, (0, 0))
        
        if actual_move == suggested_move:
            events.append(FOLLOWED_ENEMY_DIRECTION)
            followed_enemy_direction = True

    if not followed_enemy_direction and old_game_state['others'] and new_game_state['others']:
        old_nearest_enemy = min(old_game_state['others'], key=lambda e: manhattan_distance(old_position, e[3]))
        new_nearest_enemy = min(new_game_state['others'], key=lambda e: manhattan_distance(new_position, e[3]))
        
        old_distance = manhattan_distance(old_position, old_nearest_enemy[3])
        new_distance = manhattan_distance(new_position, new_nearest_enemy[3])
        
        if new_distance < old_distance:
            events.append(MOVED_TOWARDS_ENEMY)
        elif new_distance > old_distance:
            events.append(MOVED_AWAY_FROM_ENEMY)

    if e.BOMB_DROPPED in events:
        bomb_position = new_game_state['self'][3]  # Position of the agent (who just dropped the bomb)
        crate_nearby = False
        enemy_nearby = False

        # Check surroundings
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            x, y = bomb_position[0] + dx, bomb_position[1] + dy
            if new_game_state['field'][x, y] == 1:  # Crate
                crate_nearby = True
            if any(opponent[3] == (x, y) for opponent in new_game_state['others']):
                enemy_nearby = True

        if enemy_nearby:
            events.append(BOMB_PLACED_NEAR_ENEMY)
        elif crate_nearby:
            events.append(BOMB_PLACED_NEAR_CRATE)
        else:
            events.append(BOMB_PLACED_INEFFECTIVELY)

        escape_status = check_escape_routes(new_game_state, bomb_position)
        if escape_status == "UNSAFE":
            events.append(BOMB_PLACED_UNSAFELY)
        elif escape_status == "RISKY":
            events.append(BOMB_PLACED_RISKILY)

    reward = reward_from_events(self, events)

    # Store transition
    if self.random_value >= self.prob_record:
        self.transitions.append(Transition(old_features, self_action, new_features, reward))

    # Update episode statistics
    self.episode_reward += reward
    self.survival_time += 1
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1
 
    # Perform a training step
    if self.steps_since_train >= TRAIN_FREQ:
        self.train_step()
        self.steps_since_train = 0

    
def enemy_game_events_occurred(self, enemy_name: str, old_enemy_game_state: dict, enemy_action: str, new_enemy_game_state: dict, enemy_events: List[str]):    
    if self.current_step != new_enemy_game_state['step']:
        self.current_step = new_enemy_game_state['step']
        self.random_value = random.random()

    if self.random_value < self.prob_record:
        old_features = state_to_features(old_enemy_game_state)
        new_features = state_to_features(new_enemy_game_state)
        reward = reward_from_events(self, enemy_events)
        
        # Store enemy transition
        self.transitions.append(Transition(old_features, enemy_action, new_features, reward))


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

    # Terminal state Q-value update
    last_features = state_to_features(last_game_state)
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    # Update Q-value for the last action
    current_q = self.q_table[last_features][last_action]
    self.q_table[last_features][last_action] += 0.1 * (reward - current_q)

    # Add this last transition to the replay buffer
    self.transitions.append(Transition(last_features, last_action, None, reward))

    # Update prob_record for the next episode
    self.prob_record = max(0.05, RECORD_ENEMY_TRANSITIONS * (1 - last_game_state['round'] / N_TRAINING_EPSD))

    # Update epsilon 
    self.current_epsilon = max(0.025, min(1, 1.0 - last_game_state['round'] / N_TRAINING_EPSD))

    # Determine if the agent won
    won = e.SURVIVED_ROUND in events

    # Log the episode
    if last_game_state['round'] % 100 == 0:
        self.tracker.log_episode(self.episode_reward, won, self.survival_time, self.coins_collected)

    # Reset episode statistics
    self.episode_reward = 0
    self.coins_collected = 0
    self.survival_time = 0
    self.current_step = 0
    self.steps_since_train = 0

    if last_game_state['round'] == N_TRAINING_EPSD:
        self.tracker.save_current_session("q_learning_agent_v2.2 (added bomb placement quality check)")

    # Store the model
    if last_game_state['round'] == N_TRAINING_EPSD:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(dict(self.q_table), file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 10,
        e.KILLED_OPPONENT: 50,
        e.KILLED_SELF: -100,
        e.INVALID_ACTION: -5,
        e.WAITED: -1,
        #e.BOMB_DROPPED: 5,
        e.CRATE_DESTROYED: 3,
        FOLLOWED_COIN_DIRECTION: 4,
        MOVED_TOWARDS_COIN: 3,
        MOVED_AWAY_FROM_COIN: -4,
        FOLLOWED_ENEMY_DIRECTION: 2,   # Reward for following the suggested enemy direction
        MOVED_TOWARDS_ENEMY: 1.4,      # Smaller reward for any move that decreases distance to enemy
        MOVED_AWAY_FROM_ENEMY: -2,
        BOMB_PLACED_NEAR_ENEMY: 10,
        BOMB_PLACED_NEAR_CRATE: 6,
        BOMB_PLACED_INEFFECTIVELY: -3,
        BOMB_PLACED_UNSAFELY: -80,  # Big negative reward
        BOMB_PLACED_RISKILY: -7,   # Smaller negative reward
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


