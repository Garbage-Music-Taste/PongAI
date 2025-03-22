import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from RL.DQN import DQN
from RL.ReplayBuffer import ReplayBuffer


class Agent:
    '''
    state_dim: length of state vector (e.g. 5 features for Pong)
    action_dim: number of actions (e.g. 3 = [left, stay, right])
    buffer_capacity: how many past experiences to remember
    gamma: the discount factor — how much future rewards matter (close to 1 = very patient)
    lr: learning rate for the optimizer
    epsilon_*: parameters for the exploration schedule (TODO: whatever, maybe make it linear later)
    '''
    def __init__(self, state_dim, action_dim, buffer_capacity=10000, gamma=0.99, lr=1e-3,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=1000000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target net is fixed
        # fanman's inner and outer, inner is the policy net (the learner/chooser)
        # the outer is the target net which does the evaluating.
        # idk what happens if they get switched lol, hmm

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

    def select_action(self, state):
        """
            Select an action using epsilon-greedy policy.

            Args:
                state (np.ndarray): shape (state_dim,)
                policy_net (DQN): PyTorch model
                epsilon (float): probability of random action

            Returns:
                action (int): one of [0, 1, 2]
            """
        self.step_count += 1
        #self.epsilon = self.epsilon_end + (self.epsilon - self.epsilon_end) * \
            #      np.exp(-1. * self.step_count / self.epsilon_decay)
        # exp decay, simulated annealing kinda
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon_start - (self.step_count / self.epsilon_decay)
        )

        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return  # not enough data yet

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float32) # flatten first
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones) #literally bellman equation

        loss = nn.MSELoss()(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
