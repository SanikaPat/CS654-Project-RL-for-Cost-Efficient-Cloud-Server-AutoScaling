
import numpy as np
from .config import TrafficConfig


class TrafficGenerator:
    """Generates incoming requests using different traffic patterns."""

    VALID_PATTERNS = {"gradual", "daily", "spikes", "mixed"}

    def __init__(self, config=None):
        self.config = config or TrafficConfig()

        if self.config.pattern not in self.VALID_PATTERNS:
            raise ValueError(
                f"Invalid traffic pattern: {self.config.pattern}"
            )

        self.rng = np.random.default_rng(self.config.seed)
        self.time_step = 0
        self.spike_remaining = 0

    def reset(self):
        """Reset traffic simulation."""

        self.time_step = 0
        self.spike_remaining = 0

        return self.get_traffic_level(self.config.low_rate)

    def get_base_rate(self):
        """Calculate traffic before applying sudden spikes."""

        c = self.config
        progress = min(
            self.time_step / max(c.episode_length - 1, 1),
            1.0
        )

        if c.pattern == "gradual":
            rate = c.low_rate + (
                c.high_rate - c.low_rate
            ) * progress

        elif c.pattern == "daily":
            phase = 2 * np.pi * (
                self.time_step % c.daily_period
            ) / c.daily_period

            rate = c.low_rate + (
                c.high_rate - c.low_rate
            ) * (0.5 + 0.5 * np.sin(phase))

        elif c.pattern == "spikes":
            rate = c.medium_rate

        else:  # mixed
            gradual_rate = c.low_rate + (
                c.high_rate - c.low_rate
            ) * progress

            phase = 2 * np.pi * (
                self.time_step % c.daily_period
            ) / c.daily_period

            daily_factor = 0.8 + 0.4 * (
                0.5 + 0.5 * np.sin(phase)
            )

            rate = gradual_rate * daily_factor

        return rate

    def get_traffic_level(self, rate):
        """Convert traffic rate into a discrete state."""

        if rate < 20:
            return 0  # Low

        elif rate < 35:
            return 1  # Medium

        return 2  # High

    def step(self):
        """Generate traffic for one simulation time step."""

        rate = self.get_base_rate()

        # Start a sudden traffic spike
        if (
            self.spike_remaining == 0
            and self.rng.random() < self.config.spike_probability
        ):
            self.spike_remaining = self.config.spike_duration

        # Apply spike
        if self.spike_remaining > 0:
            rate *= self.config.spike_multiplier
            self.spike_remaining -= 1

        # Generate random incoming requests
        incoming_requests = int(self.rng.poisson(rate))

        traffic_level = self.get_traffic_level(rate)

        self.time_step += 1

        return traffic_level, incoming_requests, rate
