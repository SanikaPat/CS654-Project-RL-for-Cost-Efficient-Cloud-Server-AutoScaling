
class ThresholdBaseline:

    def __init__(self, min_servers=1, max_servers=10):
        self.min_servers = min_servers
        self.max_servers = max_servers

    def select_action(self, state):

        traffic, servers, queue, latency = state

        # Scale up when the system is overloaded
        if queue == 2 or latency == 2:
            if servers < self.max_servers:
                return 0

        # Scale down when demand is consistently low
        elif queue == 0 and latency == 0 and traffic == 0:
            if servers > self.min_servers:
                return 1

        # Otherwise maintain current capacity
        return 2
