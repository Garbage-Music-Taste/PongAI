import time

from PongEnv import PongEnv
from RL.Agent import Agent

state_dim = 5
action_dim = 3
agent = Agent(state_dim, action_dim)

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

    print(f"Episode {episode} | Total reward: {total_reward}")
