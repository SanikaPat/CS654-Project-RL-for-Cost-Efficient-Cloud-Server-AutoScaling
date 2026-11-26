from dataclasses import dataclass, field

@dataclass
class TrafficConfig:
    pattern: str = "mixed"

    low_rate: float = 10.0
    medium_rate: float = 25.0
    high_rate: float = 45.0

    markov_transition_matrix: list = field(
        default_factory=lambda: [
            [0.70, 0.25, 0.05],
            [0.20, 0.60, 0.20],
            [0.05, 0.25, 0.70]
        ]
    )

    spike_probability: float = 0.05
    spike_multiplier: float = 2.0
    spike_duration: int = 3

    daily_period: int = 24

    episode_length: int = 100

    seed: int | None = None


@dataclass
class CloudConfig:
    min_servers: int = 1
    max_servers: int = 10

    server_capacity: int = 10

    server_cost: float = 1.0

    latency_target: float = 1.0
    queue_limit: int = 50

    latency_penalty: float = 5.0
    dropped_request_penalty: float = 20.0

    scaling_penalty: float = 2

    episode_length: int = 100
