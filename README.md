# Pong Reinforcement Learning Agent

Train a simple reinforcement learning agent to play Pong using a low-dimensional state space.

## 🧠 State Representation
- Ball position `(x, y)`
- Ball velocity `(vx, vy)`
- Paddle 1 (agent) position
- Paddle 2 (opponent) position
- Distance to ball

In total, a 7D Vector fed into the DQN as input for predicting action values.


## 🎮 Actions
- Move paddle **up**
- Move paddle **down**
- **Do nothing**

## 🎯 Rewards
- `+10` when the agent scores
- `-5` when the opponent scores
- `+1` for hitting the ball
- `+0.01` for every live frame
- `+0.05 * (1 - (distance/max_distance))` to incentivise lining up the paddle with the ball
  
It's quite complicated and not ideal, but it yields useful results in the long run.


## 🔁 Training Loop
At each timestep:
1. Agent observes state
2. Picks an action
3. Environment updates
4. Agent receives reward

## 📦 Dependencies
- Python 3.x
- Pygame (for rendering, optional)
- PyTorch
- Matplotlib
