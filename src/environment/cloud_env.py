
from .config import CloudConfig, TrafficConfig
from .traffic_generator import TrafficGenerator


class CloudAutoScalingEnv:

    ADD_SERVER = 0
    REMOVE_SERVER = 1
    KEEP_SERVERS = 2

    def __init__(
        self,
        cloud_config=None,
        traffic_config=None,
        seed=None
    ):

        self.config = cloud_config or CloudConfig()

        self.traffic_config = traffic_config or TrafficConfig()

        if seed is not None:
            self.traffic_config.seed = seed

        self.min_servers = self.config.min_servers
        self.max_servers = self.config.max_servers

        if not 1 <= self.min_servers <= self.max_servers:
            raise ValueError("Invalid server limits")

        if self.config.server_capacity <= 0:
            raise ValueError("Server capacity must be positive")

        if self.config.queue_limit <= 0:
            raise ValueError("Queue limit must be positive")

        self.actions = [
            self.ADD_SERVER,
            self.REMOVE_SERVER,
            self.KEEP_SERVERS
        ]

        self.traffic_generator = TrafficGenerator(
            self.traffic_config
        )

        self.traffic_level = 0
        self.active_servers = 2
        self.queue_length = 0
        self.time_step = 0

    def reset(self, seed=None):
        """Reset the environment for a new episode."""

        if seed is not None:
            self.traffic_generator = TrafficGenerator(
                TrafficConfig(
                    **{
                        **self.traffic_config.__dict__,
                        "seed": seed
                    }
                )
            )

        self.traffic_level = self.traffic_generator.reset()

        self.active_servers = min(
            max(2, self.min_servers),
            self.max_servers
        )

        self.queue_length = 0
        self.time_step = 0

        return self.get_state()

    def get_state(self):
        """Convert continuous values into discrete states."""

        if self.queue_length < 10:
            queue_level = 0
        elif self.queue_length < 30:
            queue_level = 1
        else:
            queue_level = 2

        capacity = (
            self.active_servers *
            self.config.server_capacity
        )

        latency = self.queue_length / capacity

        if latency < 0.5:
            latency_level = 0
        elif latency < self.config.latency_target:
            latency_level = 1
        else:
            latency_level = 2

        return (
            self.traffic_level,
            self.active_servers,
            queue_level,
            latency_level
        )

    def step(self, action):
        """Execute one scaling action."""

        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        # Apply scaling decision
        if action == self.ADD_SERVER:
            self.active_servers = min(
                self.active_servers + 1,
                self.max_servers
            )

        elif action == self.REMOVE_SERVER:
            self.active_servers = max(
                self.active_servers - 1,
                self.min_servers
            )

        # Generate incoming traffic
        (
            self.traffic_level,
            incoming_requests,
            traffic_rate
        ) = self.traffic_generator.step()

        # Process incoming requests
        capacity = (
            self.active_servers *
            self.config.server_capacity
        )

        total_requests = (
            self.queue_length + incoming_requests
        )

        processed = min(total_requests, capacity)

        remaining_queue = total_requests - processed

        dropped = max(
            0,
            remaining_queue - self.config.queue_limit
        )

        self.queue_length = remaining_queue - dropped

        # Calculate performance metrics
        latency = self.queue_length / capacity

        latency_violation = (
            latency > self.config.latency_target
        )

        # Calculate reward
        cost = (
            self.active_servers *
            self.config.server_cost
        )

        latency_penalty = (
            max(0, latency - self.config.latency_target)
            * self.config.latency_penalty
        )

        dropped_penalty = (
            dropped *
            self.config.dropped_request_penalty
        )

        reward = (
            -cost
            -latency_penalty
            -dropped_penalty
        )

        self.time_step += 1

        done = (
            self.time_step >= self.config.episode_length
        )

        next_state = self.get_state()

        info = {
            "cost": cost,
            "latency": latency,
            "latency_violation": latency_violation,
            "dropped_requests": dropped,
            "incoming_requests": incoming_requests,
            "traffic_rate": traffic_rate,
            "processed_requests": processed,
            "active_servers": self.active_servers,
            "queue_length": self.queue_length,
            "scaling_action": action
        }

        return next_state, reward, done, info
