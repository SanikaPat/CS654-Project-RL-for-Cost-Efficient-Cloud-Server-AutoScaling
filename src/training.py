def train_agent(agent, env, num_episodes):

    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        action = agent.select_training_action(state)

        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, done, _ = env.step(action)

            next_action = agent.select_training_action(next_state)

            agent.update(state, action, reward, next_state, next_action)

            total_reward += reward
            state = next_state
            action = next_action

        agent.decay_epsilon()

        episode_rewards.append(total_reward)

    return episode_rewards
