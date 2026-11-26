# Actions used by the environment
ADD_SERVER = 0
REMOVE_SERVER = 1
KEEP_SERVERS = 2
NUM_ACTIONS = 3

# How many values each part of the state can have
NUM_TRAFFIC_LEVELS = 3 
NUM_QUEUE_LEVELS = 3 
NUM_LATENCY_LEVELS = 3


def get_num_states(max_servers):
    return (
        NUM_TRAFFIC_LEVELS
        * max_servers
        * NUM_QUEUE_LEVELS
        * NUM_LATENCY_LEVELS
    )


def state_to_index(state, max_servers):
    traffic, servers, queue, latency = state

    server_number = servers - 1

    index = traffic
    index = index * max_servers + server_number
    index = index * NUM_QUEUE_LEVELS + queue
    index = index * NUM_LATENCY_LEVELS + latency

    return index
