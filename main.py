import random
import time

from PongEnv import PongEnv

env = PongEnv(render_mode=True)

NUM_EPISODES = 5

for episode in range(NUM_EPISODES):
    print(f"Starting episode {episode + 1}")
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = random.randint(0, 2)  # 0 = left, 1 = stay, 2 = right
        next_state, reward, done = env.step(action)
        total_reward += reward
       # time.sleep(0.01)  # Optional: slow it down so we can watch

    print(f"Episode {episode + 1} finished. Total reward: {total_reward}")
