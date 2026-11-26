from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.valueIteration import ValueIterationAgent
from src.evaluation import evaluate_agent, make_env
from src.training import train_agent
from src.utilities import get_num_states, state_to_index

def test_state_index_is_unique_and_in_range():
    max_servers = 10
    seen = set()

    for traffic in range(3):
        for servers in range(1, max_servers + 1):
            for queue in range(3):
                for latency in range(3):
                    index = state_to_index(
                        (traffic, servers, queue, latency), max_servers
                    )
                    assert 0 <= index < get_num_states(max_servers)
                    seen.add(index)

    assert len(seen) == get_num_states(max_servers)


def test_q_learning_update_uses_best_next_action():
    agent = QLearningAgent(alpha=0.5, gamma=0.9, seed=0)

    state = (0, 2, 0, 0)
    next_state = (0, 3, 0, 0)

    next_row = state_to_index(next_state, agent.max_servers)
    agent.q_table[next_row][0] = 10.0

    agent.update(state, 2, -1.0, next_state, next_action=2)

    row = state_to_index(state, agent.max_servers)

    # 0.5 * (-1 + 0.9 * 10 - 0) = 4.0
    assert abs(agent.q_table[row][2] - 4.0) < 1e-9


def test_sarsa_update_uses_chosen_next_action():
    agent = SarsaAgent(alpha=0.5, gamma=0.9, seed=0)

    state = (0, 2, 0, 0)
    next_state = (0, 3, 0, 0)

    next_row = state_to_index(next_state, agent.max_servers)
    agent.q_table[next_row][0] = 10.0

    agent.update(state, 2, -1.0, next_state, next_action=2)

    row = state_to_index(state, agent.max_servers)

    # 0.5 * (-1 + 0.9 * 0 - 0) = -0.5
    assert abs(agent.q_table[row][2] - (-0.5)) < 1e-9


def test_epsilon_decays_but_stays_above_minimum():
    agent = QLearningAgent(epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.5)

    for _ in range(20):
        agent.decay_epsilon()

    assert agent.epsilon == 0.05


def test_train_agent_returns_one_reward_per_episode():
    agent = QLearningAgent(seed=0)
    env = make_env("markov", seed=0)

    rewards = train_agent(agent, env, num_episodes=5)

    assert len(rewards) == 5


def test_value_iteration_gives_valid_actions():
    agent = ValueIterationAgent(seed=0)
    env = make_env("markov", seed=0)

    agent.fit(env, num_episodes=50)

    state = env.reset()
    assert agent.select_action(state) in [0, 1, 2]


def test_evaluate_agent_returns_all_metrics():
    agent = QLearningAgent(seed=0)

    results = evaluate_agent(agent, "markov", num_episodes=2)

    for metric in ["reward", "cost", "dropped", "latency_violation_pct", "scaling_actions"]:
        assert metric in results


def test_evaluation_is_reproducible():
    agent = ValueIterationAgent(seed=0)
    agent.fit(make_env("markov", seed=0), num_episodes=50)

    first = evaluate_agent(agent, "mixed", num_episodes=3)
    second = evaluate_agent(agent, "mixed", num_episodes=3)

    assert first == second
