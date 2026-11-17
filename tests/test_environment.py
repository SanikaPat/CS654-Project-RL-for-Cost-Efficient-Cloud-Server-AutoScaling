
from src.environment.cloud_env import CloudAutoScalingEnv
from src.environment.config import CloudConfig, TrafficConfig


def test_environment_reset():

    env = CloudAutoScalingEnv()

    state = env.reset()

    assert len(state) == 4
    assert env.active_servers == 2


def test_environment_step():

    env = CloudAutoScalingEnv()

    env.reset()

    next_state, reward, done, info = env.step(2)

    assert len(next_state) == 4
    assert isinstance(reward, float)
    assert isinstance(done, bool)

    assert "cost" in info
    assert "latency" in info
    assert "dropped_requests" in info
    assert "latency_violation" in info


def test_server_limits():

    config = CloudConfig(
        min_servers=1,
        max_servers=3
    )

    env = CloudAutoScalingEnv(cloud_config=config)

    env.reset()

    for _ in range(20):
        env.step(0)

    assert env.active_servers <= 3

    for _ in range(20):
        env.step(1)

    assert env.active_servers >= 1


def test_invalid_action():

    env = CloudAutoScalingEnv()

    env.reset()

    try:
        env.step(5)
        assert False, "Invalid action should raise ValueError"
    except ValueError:
        pass


def test_episode_length():

    config = CloudConfig(episode_length=5)

    env = CloudAutoScalingEnv(cloud_config=config)

    env.reset()

    for i in range(5):
        _, _, done, _ = env.step(2)

    assert done is True


def test_reproducibility():

    env1 = CloudAutoScalingEnv(seed=42)
    env2 = CloudAutoScalingEnv(seed=42)

    state1 = env1.reset()
    state2 = env2.reset()

    assert state1 == state2

    results1 = [env1.step(2) for _ in range(10)]
    results2 = [env2.step(2) for _ in range(10)]

    assert results1 == results2
