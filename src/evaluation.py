from src.environment.cloud_env import CloudAutoScalingEnv
from src.environment.config import TrafficConfig

METRICS = [
    "reward",
    "cost",
    "dropped",
    "latency_violation_pct",
    "scaling_actions",
]


def make_env(pattern, seed):
    traffic_config = TrafficConfig(pattern=pattern, seed=seed)
    return CloudAutoScalingEnv(traffic_config=traffic_config)


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
        "latency_violation_pct": 100 * latency_violations / steps,
        "scaling_actions": scaling_actions,
    }


def evaluate_agent(agent, pattern, num_episodes=30, first_seed=1000):
    totals = {}
    for metric in METRICS:
        totals[metric] = 0.0

    for i in range(num_episodes):
        env = make_env(pattern, seed=first_seed + i)
        result = run_episode(env, agent)

        for metric in METRICS:
            totals[metric] += result[metric]

    averages = {}
    for metric in METRICS:
        averages[metric] = totals[metric] / num_episodes

    return averages
