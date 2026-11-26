import random
import numpy as np
from src.utilities import NUM_ACTIONS, get_num_states, state_to_index


class QLearningAgent:

    def __init__(
        self,
        max_servers=10,
        alpha=0.1,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.max_servers = max_servers
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.random = random.Random(seed)

        num_states = get_num_states(max_servers)
        self.q_table = np.zeros((num_states, NUM_ACTIONS))

    def select_action(self, state):
        row = state_to_index(state, self.max_servers)
        q_values = self.q_table[row]

        best_value = max(q_values)
        best_actions = []
        for action in range(NUM_ACTIONS):
            if q_values[action] == best_value:
                best_actions.append(action)

        return self.random.choice(best_actions)

    def select_training_action(self, state):
        if self.random.random() < self.epsilon:
            return self.random.randrange(NUM_ACTIONS)

        return self.select_action(state)

    def update(self, state, action, reward, next_state, next_action):
        row = state_to_index(state, self.max_servers)
        next_row = state_to_index(next_state, self.max_servers)

        best_next_value = max(self.q_table[next_row])

        target = reward + self.gamma * best_next_value
        error = target - self.q_table[row][action]

        self.q_table[row][action] += self.alpha * error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
