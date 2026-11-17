
from dataclasses import dataclass, field


@dataclass
class TrafficConfig:
    """Configuration for incoming cloud traffic."""

    # Supported traffic patterns:
    # gradual, daily, spikes, mixed, markov
    pattern: str = "mixed"

    # Average incoming requests per time step
    low_rate: float = 10.0
    medium_rate: float = 25.0
    high_rate: float = 45.0

    # Markov traffic transition probabilities
    # Rows: current traffic level
    # Columns: next traffic level
    markov_transition_matrix: list = field(
        default_factory=lambda: [
            [0.70, 0.25, 0.05],
            [0.20, 0.60, 0.20],
            [0.05, 0.25, 0.70]
        ]
    )

    # Sudden traffic spikes
    spike_probability: float = 0.05
    spike_multiplier: float = 2.0
    spike_duration: int = 3

    # Daily traffic cycle
    daily_period: int = 24

    # Episode configuration
    episode_length: int = 100

    # Random seed
    seed: int | None = None


@dataclass
class CloudConfig:
    """Configuration for the cloud server environment."""

    min_servers: int = 1
    max_servers: int = 10

    # Requests processed by one server per time step
    server_capacity: int = 10

    # Cost per active server per time step
    server_cost: float = 1.0

    # Latency and queue settings
    latency_target: float = 1.0
    queue_limit: int = 50

    # Reward penalties
    latency_penalty: float = 5.0
    dropped_request_penalty: float = 20.0

    episode_length: int = 100
