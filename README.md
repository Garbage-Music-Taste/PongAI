# Pong Reinforcement Learning Agent

Train a simple reinforcement learning agent to play Pong using a low-dimensional state space.

## 🧠 State Representation
- Ball position `(x, y)`
- Ball velocity `(vx, vy)`
- Paddle 1 (agent) position
- Paddle 2 (opponent) position

## 🎮 Actions
- Move paddle **up**
- Move paddle **down**
- **Do nothing**

## 🎯 Rewards
- `+1` when the agent scores
- `-1` when the opponent scores
- `0` otherwise

## 🔁 Training Loop
At each timestep:
1. Agent observes state
2. Picks an action
3. Environment updates
4. Agent receives reward

## 📦 Dependencies
- Python 3.x
- Pygame (for rendering, optional)
