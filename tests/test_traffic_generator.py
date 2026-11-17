
import numpy as np
import pytest

from src.environment.config import TrafficConfig
from src.environment.traffic_generator import TrafficGenerator


def test_traffic_generator_reset():

    generator = TrafficGenerator()

    level = generator.reset()

    assert level == 0
    assert generator.time_step == 0


def test_traffic_generation():

    generator = TrafficGenerator()

    generator.reset()

    level, requests, rate = generator.step()

    assert level in [0, 1, 2]
    assert requests >= 0
    assert rate > 0


@pytest.mark.parametrize(
    "pattern",
    ["gradual", "daily", "spikes", "mixed", "markov"]
)
def test_all_traffic_patterns(pattern):

    config = TrafficConfig(pattern=pattern)

    generator = TrafficGenerator(config)

    generator.reset()

    for _ in range(100):

        level, requests, rate = generator.step()

        assert level in [0, 1, 2]
        assert requests >= 0
        assert rate > 0


def test_invalid_traffic_pattern():

    config = TrafficConfig(pattern="invalid")

    with pytest.raises(ValueError):
        TrafficGenerator(config)


def test_reproducible_traffic():

    config1 = TrafficConfig(
        pattern="markov",
        seed=42
    )

    config2 = TrafficConfig(
        pattern="markov",
        seed=42
    )

    generator1 = TrafficGenerator(config1)
    generator2 = TrafficGenerator(config2)

    generator1.reset()
    generator2.reset()

    results1 = [
        generator1.step() for _ in range(100)
    ]

    results2 = [
        generator2.step() for _ in range(100)
    ]

    assert results1 == results2


def test_markov_transition_probabilities():

    config = TrafficConfig(
        pattern="markov",
        seed=42
    )

    generator = TrafficGenerator(config)

    # Force current state to Low
    generator.reset()

    observations = []

    # Collect many transitions from Low
    for _ in range(10000):

        generator.current_traffic_level = 0

        level, _, _ = generator.step()

        observations.append(level)

    observed = np.bincount(
        observations,
        minlength=3
    ) / len(observations)

    expected = np.array([0.70, 0.25, 0.05])

    assert np.allclose(
        observed,
        expected,
        atol=0.02
    )


def test_invalid_markov_transition_matrix():

    config = TrafficConfig(
        pattern="markov",
        markov_transition_matrix=[
            [0.5, 0.5, 0.5],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7]
        ]
    )

    with pytest.raises(ValueError):
        TrafficGenerator(config)
