import os
import pandas as pd
from src.agents.baseline import ThresholdBaseline
from src.agents.q_learning import QLearningAgent
from src.agents.sarsa import SarsaAgent
from src.agents.valueIteration import ValueIterationAgent
from src.evaluation import evaluate_agent, make_env
from src.plotting import plot_comparison, plot_learning_curves
from src.training import train_agent

TRAIN_PATTERNS = ["markov", "mixed"]
TEST_PATTERNS = ["markov", "gradual", "daily", "spikes", "mixed"]

TRAIN_EPISODES = 3000
VI_DATA_EPISODES = 2000
TEST_EPISODES = 30

ALPHA = 0.2
GAMMA = 0.9


BEST_SEEDS = {
    "markov": {
        "Value Iteration": 4,
        "Q-Learning": 2,
        "SARSA": 6,
    },
    "mixed": {
        "Value Iteration": 3,
        "Q-Learning": 8,
        "SARSA": 4,
    },
}

RESULTS_FOLDER = "results"


def run_experiment(train_pattern):
    print(f"\n Training on {train_pattern} traffic")

    seeds = BEST_SEEDS[train_pattern]

    # train Q-Learning and SARSA
    print("Training Q-Learning...")
    q_agent = QLearningAgent(alpha=ALPHA, gamma=GAMMA, seed=seeds["Q-Learning"])
    q_env = make_env(train_pattern, seed=seeds["Q-Learning"])
    q_rewards = train_agent(q_agent, q_env, TRAIN_EPISODES)

    print("Training SARSA...")
    sarsa_agent = SarsaAgent(alpha=ALPHA, gamma=GAMMA, seed=seeds["SARSA"])
    sarsa_env = make_env(train_pattern, seed=seeds["SARSA"])
    sarsa_rewards = train_agent(sarsa_agent, sarsa_env, TRAIN_EPISODES)

    # Value Iteration
    print("Building model and running Value Iteration...")
    vi_agent = ValueIterationAgent(gamma=GAMMA, seed=seeds["Value Iteration"])
    vi_env = make_env(train_pattern, seed=seeds["Value Iteration"])
    vi_agent.fit(vi_env, VI_DATA_EPISODES)

    # test everything
    agents = {
        "Baseline": ThresholdBaseline(),
        "Value Iteration": vi_agent,
        "Q-Learning": q_agent,
        "SARSA": sarsa_agent,
    }

    rows = []
    for agent_name in agents:
        for pattern in TEST_PATTERNS:
            print(f"Testing {agent_name} on {pattern} traffic...")
            averages = evaluate_agent(agents[agent_name], pattern, TEST_EPISODES)

            row = {"agent": agent_name, "pattern": pattern}
            row.update(averages)
            rows.append(row)

    results = pd.DataFrame(rows)

    # save and show results
    prefix = os.path.join(RESULTS_FOLDER, f"train_{train_pattern}")

    results.to_csv(prefix + "_results.csv", index=False)
    plot_learning_curves(q_rewards, sarsa_rewards, prefix + "_learning_curves.png")
    plot_comparison(results, prefix + "_comparison.png")

    print(f"\n Results (trained on {train_pattern})")
    print(results.round(2).to_string(index=False))

    print(f"\n Average over all test patterns (trained on {train_pattern})")
    summary = results.drop(columns="pattern").groupby("agent").mean()
    print(summary.round(2).to_string())


def main():
    os.makedirs(RESULTS_FOLDER, exist_ok=True)

    for train_pattern in TRAIN_PATTERNS:
        run_experiment(train_pattern)

    print(f"\nAll files were saved in the {RESULTS_FOLDER}/ folder.")


if __name__ == "__main__":
    main()