from collections import namedtuple, deque
import random
from .callbacks import ACTIONS
import pickle
from typing import List

from ..Tracker import MultiSessionPerformanceTracker
import events as e
from .callbacks import state_to_features, ACTIONS, get_direction_from_path

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 100000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
N_TRAINING_EPSD = 1000000
BATCH_SIZE = 32  # mini-batch size for training
TRAIN_FREQ = 4

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """

    # Check if we're continuing from a saved state
    if any(self.q_table.values()):
        print("Continuing training from saved state.")
    else:
        print("Starting training from scratch.")



    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode_reward = 0
    self.survival_time = 0
    self.coins_collected = 0
    self.tracker = MultiSessionPerformanceTracker()
    self.steps_since_train = 0
    self.epsilon = 1.0 if not any(self.q_table.values()) else 0.001


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
    
    self.current_step = new_game_state['step']
    self.steps_since_train += 1

    old_features = tuple(state_to_features(old_game_state))
    new_features = tuple(state_to_features(new_game_state))

    # Get the actual direction the agent moved
    old_position = old_game_state['self'][3]
    new_position = new_game_state['self'][3]

    custom_events = calculate_rewards(old_features, new_features, self_action, old_position, new_position)

    # Add custom events to the events list
    events.extend(custom_events)

    reward = reward_from_events(self, events)

    # Store transition
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

    # Update epsilon 
    self.current_epsilon = max(0.025, min(1, 1.0 - last_game_state['round'] / N_TRAINING_EPSD))
    
    # Determine if the agent won
    won = e.SURVIVED_ROUND in events

    # Update win tracking
    if not hasattr(self, 'recent_wins'):
        self.recent_wins = deque(maxlen=1000)
    self.recent_wins.append(1 if won else 0)

    # Print average win rate every 10,000 episodes
    if last_game_state['round'] % 10000 == 0:
        avg_win_rate = sum(self.recent_wins) / len(self.recent_wins)
        print(f"Episode {last_game_state['round']}: Average win rate over last 1000 episodes: {avg_win_rate:.2%}")

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
        self.tracker.save_current_session("q_learning_agent_v4.0")

    # Store the model
    if last_game_state['round'] % 50000 == 0:
        with open("my-saved-model.pt", "wb") as file:
            pickle.dump(dict(self.q_table), file)


def reward_from_events(self, events: List[str]) -> int:
    """
    *This is not a required function, but an idea to structure your code.*

    Here you can modify the rewards your agent get so as to en/discourage
    certain behavior.
    """
    game_rewards = {
        e.GOT_KILLED: -400,
        e.KILLED_SELF: -400,
        e.INVALID_ACTION: -50,
        "MOVED_TOWARDS_COIN": 20,
        "MOVED_AWAY_FROM_COIN": -15,
        "MOVED_TOWARDS_CRATE": 2,
        "MOVED_AWAY_FROM_CRATE": -8,
        "CRATE_BOMB_NOT_PLACED": -8,
        "MOVED_TOWARDS_ENEMY": 40,
        "MOVED_AWAY_FROM_ENEMY": -50,
        "IGNORED_SAFETY": -100,
        "UNNECESSARY_WAIT": -15,
        "UNNECESSARY_BOMB": -50,
        "GOOD_BOMB_PLACEMENT": 40,
        "EXCELLENT_BOMB_PLACEMENT": 250,
        "DID_NOT_PLACE_KILL_BOMB": -300,
        "FOLLOWED_ENDGAME_DIRECTIONS": 10,
        "DID_NOT_FOLLOW_ENDGAME_DIRECTIONS": -15,
    }

    reward_sum = 0
    
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]


    return reward_sum


def calculate_rewards(old_features, new_features, action, old_position, new_position):
    custom_events = []
    actual_direction = get_direction_from_path(old_position, new_position)

    def moved_as_recommended(recommended_direction):
        return actual_direction == recommended_direction

    # Reward for moving towards/away from coin or crate
    if old_features[5][1] in ['coin', 'crate']:
        recommended_direction = old_features[5][0]
        if moved_as_recommended(recommended_direction):
            custom_events.append(f"MOVED_TOWARDS_{old_features[5][1].upper()}")
        else:
            custom_events.append(f"MOVED_AWAY_FROM_{old_features[5][1].upper()}")

    # Rewards for following endgame_directions
    if old_features[5][1] in ['chasing_enemy', 'dodging_enemy']:
        recommended_direction = old_features[5][0]
        if moved_as_recommended(recommended_direction):
            custom_events.append(f"FOLLOWED_ENDGAME_DIRECTIONS")
        else:
            custom_events.append(f"DID_NOT_FOLLOW_ENDGAME_DIRECTIONS")

    # Reward for moving towards/away from enemy
    enemy_direction = old_features[6]
    if enemy_direction != 0:
        if moved_as_recommended(enemy_direction):
            custom_events.append("MOVED_TOWARDS_ENEMY")
        else:
            custom_events.append("MOVED_AWAY_FROM_ENEMY")

    # Penalty for not moving towards safety when in danger
    if old_features[5][1] == 'running_from_danger':
        recommended_direction = old_features[5][0]
        if not moved_as_recommended(recommended_direction):
            custom_events.append("IGNORED_SAFETY")

    # Penalty for unnecessary waiting
    if action == 'WAIT' and old_features[5][1] != 'wait':
        custom_events.append("UNNECESSARY_WAIT")

    if old_features[5][1] == 'reached_crate' and action != 'BOMB' and enemy_direction == 0:
        custom_events.append("CRATE_BOMB_NOT_PLACED")

    # Penalty for not placing 'kill_bomb'
    if old_features[7][1] == 'kill_bomb' and action != 'BOMB':
        custom_events.append("DID_NOT_PLACE_KILL_BOMB")

    # Rewards/Penalties for bomb placement
    if action == 'BOMB':
        if old_features[7][0] == 0:  # Bomb was not optimal
            custom_events.append("UNNECESSARY_BOMB")
        elif old_features[7][1] in ['crate_bomb', 'enemy_bomb']:
            custom_events.append("GOOD_BOMB_PLACEMENT")
        elif old_features[7][1] == 'kill_bomb':
            custom_events.append("EXCELLENT_BOMB_PLACEMENT")

    return custom_events