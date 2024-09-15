from collections import namedtuple, deque
from .callbacks import ACTIONS
import pickle
from typing import List

from ..AdvancedPerformanceTracker import RefinedPerformanceTracker
import events as e
from .callbacks import state_to_features

# This is only an example!
# Transition = namedtuple('Transition',
                        #('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
# RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...
N_TRAINING_EPSD = 1000000

# Events
# PLACEHOLDER_EVENT = "PLACEHOLDER"


def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.episode_reward = 0
    self.survival_time = 0
    self.coins_collected = 0
    self.tracker = RefinedPerformanceTracker(window_size=30)

    

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

    old_features = tuple(state_to_features(old_game_state))
    new_features = tuple(state_to_features(new_game_state))
    reward = reward_from_events(self, events)
    self.episode_reward += reward  
    self.survival_time += 1
    if e.COIN_COLLECTED in events:
        self.coins_collected += 1
    
    # Q-learning update
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor
    
    old_q = self.q_table[old_features][self_action]
    max_next_q = max(self.q_table[new_features].values(), default=0)
    new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
    
    self.q_table[old_features][self_action] = new_q


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

    alpha = 0.1  # learning rate
    old_q = self.q_table[last_features][last_action]
    new_q = old_q + alpha * (reward - old_q)

    self.q_table[last_features][last_action] = new_q

    # Update epsilon 
    self.current_epsilon = max(0.025, min(1, 1.0 - last_game_state['round'] / N_TRAINING_EPSD))

    # Determine if the agent won
    won = e.SURVIVED_ROUND in events

    # Log the episode
    if last_game_state['round'] % 100 == 0:  # Update every 100 episodes
        self.tracker.log_episode(self.episode_reward, won, self.survival_time, self.coins_collected)

    # Reset episode statistics
    self.episode_reward = 0
    self.coins_collected = 0
    self.survival_time = 0


    if last_game_state['round'] == N_TRAINING_EPSD:
        self.tracker.plot_performance()

    # Store the model
    if last_game_state['round'] % 100000 == 0:
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


