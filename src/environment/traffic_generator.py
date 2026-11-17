
import numpy as np
from .config import TrafficConfig


class TrafficGenerator:
    """Generates incoming requests using different traffic patterns."""

    VALID_PATTERNS = {
        "gradual",
        "daily",
        "spikes",
        "mixed",
        "markov"
    }

    def __init__(self, config=None):

        self.config = config or TrafficConfig()

        if self.config.pattern not in self.VALID_PATTERNS:
            raise ValueError(
                f"Invalid traffic pattern: {self.config.pattern}"
            )

        # Validate Markov transition matrix
        matrix = np.asarray(
            self.config.markov_transition_matrix,
            dtype=float
        )

        if matrix.shape != (3, 3):
            raise ValueError(
                "Markov transition matrix must be 3x3"
            )

        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ValueError(
                "Transition probabilities must be finite and non-negative"
            )

        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError(
                "Each row of the transition matrix must sum to 1"
            )

        if min(
            self.config.low_rate,
            self.config.medium_rate,
            self.config.high_rate
        ) <= 0:
            raise ValueError("Traffic rates must be positive")

        self.transition_matrix = matrix

        self.rng = np.random.default_rng(self.config.seed)

        self.time_step = 0
        self.spike_remaining = 0
        self.current_traffic_level = 0

        self.traffic_rates = [
            self.config.low_rate,
            self.config.medium_rate,
            self.config.high_rate
        ]

    def reset(self):
        """Reset traffic simulation."""

        self.time_step = 0
        self.spike_remaining = 0

        # Initial traffic state is Low
        self.current_traffic_level = 0

        return self.current_traffic_level

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

        elif c.pattern == "markov":

            # Markov traffic uses a separate generation method
            raise RuntimeError(
                "Use step() for Markov traffic generation"
            )

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

    def step_markov(self):
        """Generate traffic using a stationary Markov chain."""

        # Select next traffic level using the current level
        probabilities = self.transition_matrix[
            self.current_traffic_level
        ]

        next_level = int(
            self.rng.choice(
                [0, 1, 2],
                p=probabilities
            )
        )

        self.current_traffic_level = next_level

        # Get the average arrival rate for this level
        rate = self.traffic_rates[next_level]

        # Generate random incoming requests
        incoming_requests = int(
            self.rng.poisson(rate)
        )

        return next_level, incoming_requests, rate

    def step(self):
        """Generate traffic for one simulation time step."""

        # Handle stationary Markov traffic separately
        if self.config.pattern == "markov":

            result = self.step_markov()

            self.time_step += 1

            return result

        # Existing traffic patterns
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
        incoming_requests = int(
            self.rng.poisson(rate)
        )

        traffic_level = self.get_traffic_level(rate)

        self.time_step += 1

        return traffic_level, incoming_requests, rate
