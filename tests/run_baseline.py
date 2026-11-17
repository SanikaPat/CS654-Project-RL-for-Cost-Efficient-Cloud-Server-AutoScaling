from src.environment.cloud_env import CloudAutoScalingEnv
from src.agents.baseline import ThresholdBaseline

def run_episode(env, agent):
    state = env.reset()
    done = False

    total_reward = 0.0
    total_cost = 0.0
    total_dropped = 0
    latency_violations = 0
    scaling_actions = 0
    steps = 0

    while not done:
        action = agent.select_action(state)

        # 0 = add, 1 = remove
        if action in (0, 1):
            scaling_actions += 1

        next_state, reward, done, info = env.step(action)

        total_reward += reward
        total_cost += info["cost"]
        total_dropped += info["dropped_requests"]

        if info["latency"] > env.config.latency_target:
            latency_violations += 1

        steps += 1
        state = next_state

    return {
        "reward": total_reward,
        "cost": total_cost,
        "dropped": total_dropped,
        "latency_violation_pct": (
            100 * latency_violations / steps
        ),
        "scaling_actions": scaling_actions,
    }


def run_baseline_experiment(num_episodes=20):
    results = []

    for episode in range(num_episodes):
        env = CloudAutoScalingEnv()

        agent = ThresholdBaseline(
            min_servers=env.min_servers,
            max_servers=env.max_servers,
        )

        result = run_episode(env, agent)
        results.append(result)

        print(
            f"Episode {episode + 1:2d}: "
            f"Reward={result['reward']:.2f}, "
            f"Cost={result['cost']:.2f}, "
            f"Dropped={result['dropped']}, "
            f"Latency Violations="
            f"{result['latency_violation_pct']:.2f}%, "
            f"Scaling Actions={result['scaling_actions']}"
        )

    print("\n========== BASELINE RESULTS ==========")

    metrics = [
        "reward",
        "cost",
        "dropped",
        "latency_violation_pct",
        "scaling_actions",
    ]

    for metric in metrics:
        average = sum(r[metric] for r in results) / len(results)

        print(f"Average {metric}: {average:.2f}")


if __name__ == "__main__":
    run_baseline_experiment()