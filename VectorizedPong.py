from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch


@dataclass(frozen=True)
class PongConfig:
    width: float = 400.0
    height: float = 300.0
    paddle_length: float = 70.0
    paddle_height: float = 10.0
    paddle_speed: float = 10.0
    opponent_speed: float = 5.0
    ball_radius: float = 5.0
    ball_speed: float = 6.0
    max_ball_speed: float = 8.0
    max_episode_steps: int = 350


class VectorizedPong:
    """Torch-native, headless Pong environments executed in one batch.

    The environment and replay buffer can live on the GPU, avoiding the
    one-frame-at-a-time Python/Pygame bottleneck in the original trainer.
    One episode is one rally: score, concede, or hit the time limit.
    """

    STATE_DIM = 7
    ACTION_DIM = 3

    def __init__(
        self,
        num_envs: int,
        device: torch.device | str,
        seed: int = 0,
        config: Optional[PongConfig] = None,
        reward_shaping: bool = True,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")

        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.config = config or PongConfig()
        self.reward_shaping = reward_shaping
        self.dtype = torch.float32

        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

        shape = (self.num_envs,)
        self.paddle1_x = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.paddle2_x = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.ball_x = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.ball_y = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.ball_vx = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.ball_vy = torch.empty(shape, device=self.device, dtype=self.dtype)
        self.episode_steps = torch.empty(shape, device=self.device, dtype=torch.int32)
        self.agent_hits = torch.empty(shape, device=self.device, dtype=torch.int32)

        self._all = torch.ones(shape, device=self.device, dtype=torch.bool)
        self.reset()

    @property
    def opponent_speed(self) -> float:
        return self.config.opponent_speed

    def set_opponent_speed(self, speed: float) -> None:
        if speed < 0:
            raise ValueError("opponent speed must be non-negative")
        self.config = PongConfig(**{**self.config.__dict__, "opponent_speed": float(speed)})

    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            mask = self._all
        else:
            mask = mask.to(device=self.device, dtype=torch.bool)
            if mask.shape != (self.num_envs,):
                raise ValueError(f"mask must have shape ({self.num_envs},)")

        cfg = self.config
        centered = cfg.width / 2.0 - cfg.paddle_length / 2.0
        self.paddle1_x[mask] = centered
        self.paddle2_x[mask] = centered
        self.ball_x[mask] = cfg.width / 2.0
        self.ball_y[mask] = cfg.height / 2.0

        # Angle from vertical. This keeps starts varied without near-horizontal
        # trajectories that generate very long, low-information episodes.
        phi = torch.empty(self.num_envs, device=self.device, dtype=self.dtype).uniform_(
            math.pi / 6.0,
            math.pi / 3.0,
            generator=self.generator,
        )
        horizontal_sign = torch.randint(
            0, 2, (self.num_envs,), device=self.device, generator=self.generator
        ).to(self.dtype) * 2.0 - 1.0
        vertical_sign = torch.randint(
            0, 2, (self.num_envs,), device=self.device, generator=self.generator
        ).to(self.dtype) * 2.0 - 1.0

        new_vx = cfg.ball_speed * torch.sin(phi) * horizontal_sign
        new_vy = cfg.ball_speed * torch.cos(phi) * vertical_sign
        self.ball_vx[mask] = new_vx[mask]
        self.ball_vy[mask] = new_vy[mask]
        self.episode_steps[mask] = 0
        self.agent_hits[mask] = 0
        return self.state()

    def state(self) -> torch.Tensor:
        cfg = self.config
        paddle1_center = self.paddle1_x + cfg.paddle_length / 2.0
        paddle2_center = self.paddle2_x + cfg.paddle_length / 2.0
        return torch.stack(
            (
                self.ball_x / cfg.width,
                self.ball_y / cfg.height,
                self.ball_vx / cfg.max_ball_speed,
                self.ball_vy / cfg.max_ball_speed,
                paddle1_center / cfg.width,
                paddle2_center / cfg.width,
                (paddle1_center - self.ball_x) / cfg.width,
            ),
            dim=1,
        )

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance all environments by one frame.

        Returns:
            next_states: state after the action, before caller resets done envs
            rewards: shaped training reward
            dones: episode completion mask
            wins: agent-scored mask
            agent_hit: agent returned the ball this frame
        """
        actions = actions.to(device=self.device, dtype=torch.long).flatten()
        if actions.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        cfg = self.config
        self.episode_steps += 1

        old_distance = torch.abs(
            self.paddle1_x + cfg.paddle_length / 2.0 - self.ball_x
        )

        movement = torch.zeros_like(self.paddle1_x)
        movement = torch.where(actions == 0, -cfg.paddle_speed, movement)
        movement = torch.where(actions == 2, cfg.paddle_speed, movement)
        self.paddle1_x.add_(movement).clamp_(0.0, cfg.width - cfg.paddle_length)

        opponent_center = self.paddle2_x + cfg.paddle_length / 2.0
        opponent_error = self.ball_x - opponent_center
        opponent_move = torch.where(
            torch.abs(opponent_error) > 2.0,
            torch.sign(opponent_error) * cfg.opponent_speed,
            torch.zeros_like(opponent_error),
        )
        self.paddle2_x.add_(opponent_move).clamp_(
            0.0, cfg.width - cfg.paddle_length
        )

        previous_y = self.ball_y.clone()
        self.ball_x.add_(self.ball_vx)
        self.ball_y.add_(self.ball_vy)

        hit_left_wall = self.ball_x <= cfg.ball_radius
        hit_right_wall = self.ball_x >= cfg.width - cfg.ball_radius
        self.ball_x.clamp_(cfg.ball_radius, cfg.width - cfg.ball_radius)
        self.ball_vx[hit_left_wall] = torch.abs(self.ball_vx[hit_left_wall])
        self.ball_vx[hit_right_wall] = -torch.abs(self.ball_vx[hit_right_wall])

        rewards = torch.full(
            (self.num_envs,), -0.001, device=self.device, dtype=self.dtype
        )

        bottom_y = cfg.height - 20.0
        crossed_bottom = (
            (self.ball_vy > 0.0)
            & (previous_y + cfg.ball_radius < bottom_y)
            & (self.ball_y + cfg.ball_radius >= bottom_y)
        )
        agent_hit = (
            crossed_bottom
            & (self.ball_x >= self.paddle1_x)
            & (self.ball_x <= self.paddle1_x + cfg.paddle_length)
        )
        self.ball_y[agent_hit] = bottom_y - cfg.ball_radius
        offset = (
            self.ball_x[agent_hit]
            - (self.paddle1_x[agent_hit] + cfg.paddle_length / 2.0)
        ) / (cfg.paddle_length / 2.0)
        self.ball_vx[agent_hit] += 1.5 * offset
        self.ball_vy[agent_hit] = -torch.abs(self.ball_vy[agent_hit])
        self._clamp_ball_speed(agent_hit)
        self.agent_hits[agent_hit] += 1
        rewards[agent_hit] += 1.0

        top_y = 10.0
        top_surface = top_y + cfg.paddle_height
        crossed_top = (
            (self.ball_vy < 0.0)
            & (previous_y - cfg.ball_radius > top_surface)
            & (self.ball_y - cfg.ball_radius <= top_surface)
        )
        opponent_hit = (
            crossed_top
            & (self.ball_x >= self.paddle2_x)
            & (self.ball_x <= self.paddle2_x + cfg.paddle_length)
        )
        self.ball_y[opponent_hit] = top_surface + cfg.ball_radius
        offset = (
            self.ball_x[opponent_hit]
            - (self.paddle2_x[opponent_hit] + cfg.paddle_length / 2.0)
        ) / (cfg.paddle_length / 2.0)
        self.ball_vx[opponent_hit] += 1.5 * offset
        self.ball_vy[opponent_hit] = torch.abs(self.ball_vy[opponent_hit])
        self._clamp_ball_speed(opponent_hit)

        if self.reward_shaping:
            new_distance = torch.abs(
                self.paddle1_x + cfg.paddle_length / 2.0 - self.ball_x
            )
            toward_agent = (self.ball_vy > 0.0) & ~agent_hit
            alignment = 1.0 - torch.clamp(new_distance / (cfg.width / 2.0), 0.0, 1.0)
            improvement = (old_distance - new_distance) / cfg.paddle_speed
            rewards += toward_agent.to(self.dtype) * (
                0.02 * alignment + 0.02 * improvement
            )

        wins = self.ball_y < -cfg.ball_radius
        losses = self.ball_y > cfg.height + cfg.ball_radius
        timeouts = self.episode_steps >= cfg.max_episode_steps
        dones = wins | losses | timeouts

        rewards[wins] += 2.0
        rewards[losses] -= 1.0
        rewards[timeouts & ~wins & ~losses] -= 0.1

        return self.state(), rewards, dones, wins, agent_hit

    def _clamp_ball_speed(self, mask: torch.Tensor) -> None:
        cfg = self.config
        speed = torch.hypot(self.ball_vx[mask], self.ball_vy[mask])
        scale = torch.clamp(cfg.max_ball_speed / speed.clamp_min(1e-6), max=1.0)
        self.ball_vx[mask] *= scale
        self.ball_vy[mask] *= scale
