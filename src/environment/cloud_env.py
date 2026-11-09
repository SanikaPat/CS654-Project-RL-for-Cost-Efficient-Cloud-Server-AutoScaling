
import numpy as np


class CloudAutoScalingEnv:

    def __init__(
        self,
        min_servers=1,
        max_servers=10,
        server_capacity=10,
        server_cost=1.0,
        latency_target=1.0
    ):
        self.min_servers = min_servers
        self.max_servers = max_servers
        self.server_capacity = server_capacity
        self.server_cost = server_cost
        self.latency_target = latency_target

        self.actions = [0, 1, 2]

        self.traffic_level = 0
        self.active_servers = 2
        self.queue_length = 0
        self.time_step = 0

    def reset(self):
        """Reset the simulation."""

        self.traffic_level = 0
        self.active_servers = 2
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

        capacity = self.active_servers * self.server_capacity

        if capacity <= 0:
            latency = float("inf")
        else:
            latency = self.queue_length / capacity

        if latency < 0.5:
            latency_level = 0
        elif latency < 1.0:
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
            raise ValueError("Invalid action")

        # Apply scaling decision
        if action == 0:
            self.active_servers = min(
                self.active_servers + 1,
                self.max_servers
            )

        elif action == 1:
            self.active_servers = max(
                self.active_servers - 1,
                self.min_servers
            )

        # Generate incoming traffic
        traffic_rates = [10, 25, 45]
        incoming_requests = np.random.poisson(
            traffic_rates[self.traffic_level]
        )

        # Process requests
        capacity = self.active_servers * self.server_capacity

        total_requests = self.queue_length + incoming_requests

        processed = min(total_requests, capacity)
        self.queue_length = total_requests - processed

        dropped = max(0, self.queue_length - 50)
        self.queue_length -= dropped

        latency = self.queue_length / max(capacity, 1)

        # Calculate reward
        cost = self.active_servers * self.server_cost

        latency_penalty = max(
            0,
            latency - self.latency_target
        ) * 5

        dropped_penalty = dropped * 20

        reward = -cost - latency_penalty - dropped_penalty

        # Change traffic level
        self.time_step += 1

        if np.random.random() < 0.2:
            self.traffic_level = np.random.choice([0, 1, 2])

        next_state = self.get_state()

        done = self.time_step >= 100

        info = {
            "cost": cost,
            "latency": latency,
            "dropped_requests": dropped,
            "active_servers": self.active_servers
        }

        return next_state, reward, done, info
