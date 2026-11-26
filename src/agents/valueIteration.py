import random
import numpy as np
from src.utilities import KEEP_SERVERS, NUM_ACTIONS, get_num_states, state_to_index

UNSEEN_PENALTY = -100.0

class ValueIterationAgent:
    def __init__(self, max_servers=10, gamma=0.95, theta=0.01, max_iterations=1000, seed=None):
        self.max_servers = max_servers
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.random = random.Random(seed)

        num_states = get_num_states(max_servers)


        self.transition_counts = np.zeros((num_states, NUM_ACTIONS, num_states))
        self.reward_sums = np.zeros((num_states, NUM_ACTIONS))
        self.visits = np.zeros((num_states, NUM_ACTIONS))


        self.transition_probs = np.zeros((num_states, NUM_ACTIONS, num_states))
        self.avg_rewards = np.zeros((num_states, NUM_ACTIONS))


        self.values = np.zeros(num_states)
        self.policy = np.full(num_states, KEEP_SERVERS)

    def collect_data(self, env, num_episodes):
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = self.random.randrange(NUM_ACTIONS)
                next_state, reward, done, info = env.step(action)

                row = state_to_index(state, self.max_servers)
                next_row = state_to_index(next_state, self.max_servers)

                self.transition_counts[row][action][next_row] += 1
                self.reward_sums[row][action] += reward
                self.visits[row][action] += 1

                state = next_state

    def build_model(self):
        num_states = get_num_states(self.max_servers)

        for row in range(num_states):
            for action in range(NUM_ACTIONS):
                times_seen = self.visits[row][action]

                if times_seen > 0:
                    self.transition_probs[row][action] = (
                        self.transition_counts[row][action] / times_seen
                    )
                    self.avg_rewards[row][action] = (
                        self.reward_sums[row][action] / times_seen
                    )
                else:
                    self.transition_probs[row][action][row] = 1.0
                    self.avg_rewards[row][action] = UNSEEN_PENALTY

    def run_value_iteration(self):
        values = np.zeros(get_num_states(self.max_servers))

        for iteration in range(self.max_iterations):
            expected_next_values = np.dot(self.transition_probs, values)
            q_values = self.avg_rewards + self.gamma * expected_next_values

            new_values = q_values.max(axis=1)

            biggest_change = np.max(np.abs(new_values - values))
            values = new_values

            if biggest_change < self.theta:
                break

        self.values = values

        self.policy = q_values.argmax(axis=1)

        for row in range(len(self.policy)):
            if self.visits[row].sum() == 0:
                self.policy[row] = KEEP_SERVERS

    def fit(self, env, num_episodes=2000):
        self.collect_data(env, num_episodes)
        self.build_model()
        self.run_value_iteration()

    def select_action(self, state):
        row = state_to_index(state, self.max_servers)
        return int(self.policy[row])
