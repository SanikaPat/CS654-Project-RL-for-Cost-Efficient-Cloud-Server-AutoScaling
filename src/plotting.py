import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_average(values, window):
    weights = np.ones(window) / window
    return np.convolve(values, weights, mode="valid")


def plot_learning_curves(q_rewards, sarsa_rewards, filename, window=50):
    plt.figure(figsize=(8, 5))

    plt.plot(moving_average(q_rewards, window), label="Q-Learning")
    plt.plot(moving_average(sarsa_rewards, window), label="SARSA")

    plt.xlabel("Training episode")
    plt.ylabel(f"Total reward (average of {window} episodes)")
    plt.title("Learning curves (higher is better)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison(results, filename):
    metrics = [
        ("cost", "Average cost (lower is better)"),
        ("latency_violation_pct", "Latency violations % (lower is better)"),
        ("dropped", "Dropped requests (lower is better)"),
        ("scaling_actions", "Scaling actions (lower is more stable)"),
    ]

    agents = list(results["agent"].unique())
    patterns = list(results["pattern"].unique())

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    bar_width = 0.8 / len(agents)

    for i in range(len(metrics)):
        column, title = metrics[i]
        axis = axes[i]

        for j in range(len(agents)):
            agent_name = agents[j]
            agent_rows = results[results["agent"] == agent_name]

            heights = []
            for pattern in patterns:
                row = agent_rows[agent_rows["pattern"] == pattern]
                heights.append(row[column].values[0])

            positions = np.arange(len(patterns)) + j * bar_width
            axis.bar(positions, heights, width=bar_width, label=agent_name)

        axis.set_xticks(np.arange(len(patterns)) + bar_width * (len(agents) - 1) / 2)
        axis.set_xticklabels(patterns)
        axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.3)

    axes[0].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
