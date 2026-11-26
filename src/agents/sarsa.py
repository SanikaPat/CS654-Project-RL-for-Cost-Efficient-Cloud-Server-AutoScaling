from src.agents.q_learning import QLearningAgent
from src.utilities import state_to_index


class SarsaAgent(QLearningAgent):

    def update(self, state, action, reward, next_state, next_action):
        row = state_to_index(state, self.max_servers)
        next_row = state_to_index(next_state, self.max_servers)

        next_value = self.q_table[next_row][next_action]

        target = reward + self.gamma * next_value
        error = target - self.q_table[row][action]

        self.q_table[row][action] += self.alpha * error
