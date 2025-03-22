import time

import matplotlib

from PongEnv import PongEnv
from RL.Agent import Agent


import matplotlib
matplotlib.use('Agg')  # For headless / non-interactive plotting
import matplotlib.pyplot as plt


#plt.ion()
episode_rewards = []


agent = Agent(PongEnv.STATE_DIM, PongEnv.ACTION_DIM)

NUM_EPISODES = 1000
BATCH_SIZE = 64
TARGET_UPDATE_EVERY = 100

for episode in range(NUM_EPISODES):
    render = (episode % TARGET_UPDATE_EVERY == 0)
    env = PongEnv(render_mode=render, episode_num=episode if render else None)

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step(BATCH_SIZE)
        state = next_state
        total_reward += reward

    if episode % TARGET_UPDATE_EVERY == 0:
        agent.update_target()

    print(f"Episode {episode} | Total reward: {total_reward} | Variability: {agent.epsilon}")
    episode_rewards.append(total_reward)

    # Plot every 50 episodes
    if episode % 50 == 0 and episode > 0:
        plt.clf()
        plt.plot(episode_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("Total Reward Over Time")
        plt.savefig(f"rewards_ep{episode}.png")

plt.ioff()
plt.show()

