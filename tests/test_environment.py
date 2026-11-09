
from src.environment.cloud_env import CloudAutoScalingEnv


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
    assert isinstance(reward, (int, float))
    assert isinstance(done, bool)
    assert "cost" in info


def test_server_limits():
    env = CloudAutoScalingEnv(
        min_servers=1,
        max_servers=3
    )

    env.reset()

    for _ in range(20):
        env.step(0)

    assert env.active_servers <= 3

    for _ in range(20):
        env.step(1)

    assert env.active_servers >= 1
