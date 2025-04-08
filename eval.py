import torch
from PongEnv import PongEnv
from RL.Agent import Agent

# Create environment and agent
env = PongEnv(render_mode=True)
agent = Agent(PongEnv.STATE_DIM, PongEnv.ACTION_DIM)

# Load the trained model
agent.policy_net.load_state_dict(torch.load("trained.pth"))
agent.policy_net.eval()  # Set to evaluation mode
agent.epsilon = 0.0  # Ensure deterministic policy (no exploration)

num_episodes = 1
total_wins = 0

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = agent.policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        next_state, reward, done, win = env.step(action)
        state = next_state
        total_reward += reward

    total_wins += win
    print(f"Episode {episode} | Total reward: {total_reward} | Win: {win}")

print(f"Win rate: {total_wins / num_episodes:.2%}")