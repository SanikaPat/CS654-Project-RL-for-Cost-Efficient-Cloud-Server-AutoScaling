class ThresholdBaseline:
    """
    Simple rule-based autoscaling baseline.

    Actions:
        0 = Add server
        1 = Remove server
        2 = Keep current server count
    """

    def __init__(self, min_servers=1, max_servers=10):
        self.min_servers = min_servers
        self.max_servers = max_servers

    def select_action(self, state):
        traffic, servers, queue, latency = state

        # Scale up if overloaded
        if queue == 2 or latency == 2:
            if servers < self.max_servers:
                return 0

        # Scale down if demand is low
        if traffic == 0 and queue == 0 and latency == 0:
            if servers > self.min_servers:
                return 1

        # Otherwise keep current server count
        return 2