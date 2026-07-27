from __future__ import annotations

import argparse
import sys

import pygame
import torch

from VectorizedPong import PongConfig, VectorizedPong
from RL.VectorDQN import DuelingDQN


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch a trained vectorized Pong agent.")
    parser.add_argument("--model", default="pong_vectorized.pth")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    policy = DuelingDQN(
        checkpoint["state_dim"],
        checkpoint["action_dim"],
        checkpoint["hidden_dim"],
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    config = PongConfig(**checkpoint.get("pong_config", {}))
    env = VectorizedPong(
        num_envs=1,
        device="cpu",
        seed=args.seed,
        config=config,
        reward_shaping=False,
    )
    state = env.state()

    pygame.init()
    screen = pygame.display.set_mode((int(config.width), int(config.height)))
    pygame.display.set_caption("Vectorized Pong DQN")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    wins = losses = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.inference_mode():
            action = policy(state).argmax(dim=1)
        next_state, _, done, scored, _ = env.step(action)

        if bool(done.item()):
            wins += int(scored.item())
            losses += int(not scored.item())
            env.reset(done)
            state = env.state()
        else:
            state = next_state

        screen.fill("black")
        pygame.draw.line(
            screen,
            "white",
            (0, int(config.height / 2)),
            (int(config.width), int(config.height / 2)),
            1,
        )
        pygame.draw.rect(
            screen,
            "white",
            pygame.Rect(
                int(env.paddle1_x.item()),
                int(config.height - 20),
                int(config.paddle_length),
                int(config.paddle_height),
            ),
        )
        pygame.draw.rect(
            screen,
            "white",
            pygame.Rect(
                int(env.paddle2_x.item()),
                10,
                int(config.paddle_length),
                int(config.paddle_height),
            ),
        )
        pygame.draw.circle(
            screen,
            "white",
            (int(env.ball_x.item()), int(env.ball_y.item())),
            int(config.ball_radius),
        )
        text = font.render(f"Agent {wins} - {losses} Opponent", True, "white")
        screen.blit(text, (10, int(config.height / 2 + 10)))
        pygame.display.flip()
        clock.tick(args.fps)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
