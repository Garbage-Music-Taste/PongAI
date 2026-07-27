from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    """Small dueling network for the seven-dimensional Pong state."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden_dim, 1)
        self.advantage = nn.Linear(hidden_dim, action_dim)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.backbone(states)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DeviceReplayBuffer:
    """Preallocated replay buffer stored directly on the training device."""

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: torch.device | str,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")

        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.states = torch.empty(
            (capacity, state_dim), device=self.device, dtype=torch.float32
        )
        self.actions = torch.empty(capacity, device=self.device, dtype=torch.long)
        self.rewards = torch.empty(capacity, device=self.device, dtype=torch.float32)
        self.next_states = torch.empty(
            (capacity, state_dim), device=self.device, dtype=torch.float32
        )
        self.dones = torch.empty(capacity, device=self.device, dtype=torch.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    @torch.no_grad()
    def add_batch(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        batch_size = states.shape[0]
        if batch_size > self.capacity:
            states = states[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_states = next_states[-self.capacity :]
            dones = dones[-self.capacity :]
            batch_size = self.capacity

        indices = (
            torch.arange(batch_size, device=self.device, dtype=torch.long)
            + self.position
        ) % self.capacity
        self.states.index_copy_(0, indices, states)
        self.actions.index_copy_(0, indices, actions)
        self.rewards.index_copy_(0, indices, rewards)
        self.next_states.index_copy_(0, indices, next_states)
        self.dones.index_copy_(0, indices, dones.to(torch.float32))

        self.position = (self.position + batch_size) % self.capacity
        self.size = min(self.capacity, self.size + batch_size)

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.size < batch_size:
            raise ValueError("not enough transitions in replay buffer")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )
