
from dataclasses import dataclass


@dataclass
class TrafficConfig:
    """Configuration for incoming cloud traffic."""

    # Traffic generation pattern:
    # gradual, daily, spikes, or mixed
    pattern: str = "mixed"

    # Average incoming requests per time step
    low_rate: float = 10.0
    medium_rate: float = 25.0
    high_rate: float = 45.0

    # Sudden traffic spikes
    spike_probability: float = 0.05
    spike_multiplier: float = 2.0
    spike_duration: int = 3

    # Daily traffic cycle
    daily_period: int = 24

    # Episode configuration
    episode_length: int = 100

    # Random seed for reproducibility
    seed: int | None = None


@dataclass
class CloudConfig:
    """Configuration for the cloud server environment."""

    min_servers: int = 1
    max_servers: int = 10

    # Requests processed by one server per time step
    server_capacity: int = 10

    # Cost of keeping one server active per time step
    server_cost: float = 1.0

    # Latency and queue settings
    latency_target: float = 1.0
    queue_limit: int = 50

    # Reward penalties
    latency_penalty: float = 5.0
    dropped_request_penalty: float = 20.0

    episode_length: int = 100
