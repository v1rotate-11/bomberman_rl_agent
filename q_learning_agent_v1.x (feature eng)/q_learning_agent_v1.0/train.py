from collections import namedtuple, deque
from .callbacks import ACTIONS

import pickle
from typing import List

#from .performance_tracking import tracker
from ..PerformanceTracker import SimplePerformanceTracker
import events as e
from .callbacks import state_to_features

# This is only an example!
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters -- DO modify
TRANSITION_HISTORY_SIZE = 50000  # keep only ... last transitions
RECORD_ENEMY_TRANSITIONS = 1.0  # record enemy transitions with probability ...

# Events
PLACEHOLDER_EVENT = "PLACEHOLDER"
tracker = SimplePerformanceTracker()

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    # Example: Setup an array that will note transition tuples
    # (s, a, r, s')
    self.transitions = deque(maxlen=TRANSITION_HISTORY_SIZE)
    self.episode_reward = 0


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
    self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')
    self.logger.info(f"GAME EVENT: {events}")

    old_features = tuple(state_to_features(old_game_state))
    new_features = tuple(state_to_features(new_game_state))
    reward = reward_from_events(self, events)
    self.episode_reward += reward  
    

    # Q-learning update
    alpha = 0.1  # learning rate
    gamma = 0.9  # discount factor
    
    old_q = self.q_table.get(tuple(old_features), {}).get(self_action, 0)
    

    
    best_next_action = max(ACTIONS, key=lambda a: self.q_table.get(new_features, {}).get(a, 0))
    max_next_q = self.q_table.get(new_features, {}).get(best_next_action, 0)
    new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
    
    
    if old_features not in self.q_table:
        self.q_table[old_features] = {}
    self.q_table[old_features][str(self_action)] = new_q


    self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward))

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

    self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')

    # Terminal state Q-value update
    last_features = tuple(state_to_features(last_game_state))
    reward = reward_from_events(self, events)
    self.episode_reward += reward

    alpha = 0.1  # learning rate
    old_q = self.q_table.get(last_features, {}).get(last_action, 0)
    new_q = old_q + alpha * (reward - old_q)

    if last_features not in self.q_table:
        self.q_table[last_features] = {}
    self.q_table[last_features][last_action] = new_q

    self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward))

    tracker.log_episode(self.episode_reward)
    self.episode_reward = 0

    if last_game_state['round'] % 100 == 0:
        tracker.plot_average_score()

    # Store the model
    with open("my-saved-model.pt", "wb") as file:
        pickle.dump(self.q_table, file)


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
