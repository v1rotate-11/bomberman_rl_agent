from collections import namedtuple, deque
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
        self.tracker.save_current_session("q_learning_agent_v1.5 (added extensive adjacent tile info)")

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
        e.BOMB_DROPPED: 5,
        e.CRATE_DESTROYED: 2
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum


