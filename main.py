import select
import sys
import time

import matplotlib
import numpy as np
import torch

from PongEnv import PongEnv
from RL.Agent import Agent

import matplotlib

matplotlib.use('Agg')  # For headless / non-interactive plotting
import matplotlib.pyplot as plt

# plt.ion()
episode_rewards = []

agent = Agent(PongEnv.STATE_DIM, PongEnv.ACTION_DIM)
# agent.policy_net.load_state_dict(torch.load("pong_model_trained.pth"))
# agent.policy_net.train()
# agent.epsilon_start = 0.1  # always choose best move


NUM_EPISODES = 10000
BATCH_SIZE = 64
TARGET_UPDATE_EVERY = 25

win_count = 0
wins = []
win_rates = []


def check_user_input():
    # Non-blocking check for "" typed in the terminal
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        line = sys.stdin.readline().strip()
        return line.lower() == 'a'
    return False

manual_render = False

for episode in range(0, NUM_EPISODES):
    if check_user_input():
        manual_render = not manual_render  # toggle rendering

    env = PongEnv(render_mode=manual_render, episode_num=episode if manual_render else None)
    if manual_render:
        manual_render = not manual_render # toggle back

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, win = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train_step(BATCH_SIZE)
        state = next_state
        total_reward += reward

    if episode % TARGET_UPDATE_EVERY == 0:
        agent.update_target()

    win_count += win
    wins.append(win)

    awr = sum(wins[-100:]) / 100 if len(wins) > 100 else win_count / (1 + episode)
    win_rates.append(awr)

    print(
        f"Episode {episode} | Total reward: {total_reward} | Variability: {agent.epsilon} | WR: {win_count / (1 + episode)} | AWR: {awr} | Avg Q: {agent.avg_q_value:.4f}")
    episode_rewards.append(total_reward)

    # Plot every 100 episodes
    if episode % 100 == 0 and episode > 0:
        plt.clf()
        plt.subplot(2, 1, 1)
        if len(episode_rewards) >= 10:
            ma = np.convolve(episode_rewards, np.ones(10) / 10, mode='valid')
            plt.plot(range(9, len(episode_rewards)), ma, label='10-ep reward avg')
        plt.plot(episode_rewards, alpha=0.4, label='Reward')
        plt.legend()
        plt.ylabel("Reward")
        plt.title("Training Progress")

        plt.subplot(2, 1, 2)
        plt.plot(win_rates, color='green', label='Win rate (AWR)')
        plt.ylabel("Win Rate")
        plt.xlabel("Episode")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"rewards_ep{episode}.png")
        torch.save(agent.policy_net.state_dict(), "pong_model.pth")

plt.ioff()
plt.show()
