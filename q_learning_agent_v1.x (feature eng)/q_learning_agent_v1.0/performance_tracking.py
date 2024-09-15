import matplotlib.pyplot as plt
from collections import deque

class PerformanceTracker:
    def __init__(self, window_size=100):
        self.scores = deque(maxlen=window_size)
        self.rewards = deque(maxlen=window_size)
        self.coins_collected = deque(maxlen=window_size)
        self.opponents_killed = deque(maxlen=window_size)
        self.deaths = deque(maxlen=window_size)
        self.avg_q_values = deque(maxlen=window_size)

    def add_data(self, score, total_reward, coins, kills, died, avg_q_value):
        self.scores.append(score)
        self.rewards.append(total_reward)
        self.coins_collected.append(coins)
        self.opponents_killed.append(kills)
        self.deaths.append(int(died))
        self.avg_q_values.append(avg_q_value)

    def plot_performance(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(3, 2, 1)
        plt.plot(list(self.scores))
        plt.title('Average Score')

        plt.subplot(3, 2, 2)
        plt.plot(list(self.rewards))
        plt.title('Average Reward')

        plt.subplot(3, 2, 3)
        plt.plot(list(self.coins_collected))
        plt.title('Coins Collected')

        plt.subplot(3, 2, 4)
        plt.plot(list(self.opponents_killed))
        plt.title('Opponents Killed')

        plt.subplot(3, 2, 5)
        plt.plot(list(self.deaths))
        plt.title('Deaths')

        plt.subplot(3, 2, 6)
        plt.plot(list(self.avg_q_values))
        plt.title('Average Q-Value')

        plt.tight_layout()
        plt.savefig('performance_plot.png')
        plt.close()

tracker = PerformanceTracker()
