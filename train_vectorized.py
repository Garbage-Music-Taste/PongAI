from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from VectorizedPong import PongConfig, VectorizedPong
from RL.VectorDQN import DeviceReplayBuffer, DuelingDQN


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is false")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_at(
    transitions: int,
    start: float,
    end: float,
    decay_transitions: int,
) -> float:
    fraction = min(max(transitions / max(decay_transitions, 1), 0.0), 1.0)
    return start + fraction * (end - start)


@torch.no_grad()
def evaluate(
    policy: DuelingDQN,
    device: torch.device,
    episodes: int,
    num_envs: int,
    seed: int,
    opponent_speed: float,
) -> dict[str, float]:
    policy.eval()
    eval_envs = min(num_envs, episodes)
    env = VectorizedPong(
        num_envs=eval_envs,
        device=device,
        seed=seed,
        config=PongConfig(opponent_speed=opponent_speed),
        reward_shaping=False,
    )
    states = env.state()

    completed = 0
    wins = 0
    returns = 0
    total_hits = 0
    total_length = 0

    while completed < episodes:
        actions = policy(states).argmax(dim=1)
        next_states, _, dones, scored, _ = env.step(actions)

        done_count = int(dones.sum().item())
        if done_count:
            remaining = episodes - completed
            done_indices = torch.nonzero(dones, as_tuple=False).flatten()[:remaining]
            wins += int(scored[done_indices].sum().item())
            hits = env.agent_hits[done_indices]
            returns += int((hits > 0).sum().item())
            total_hits += int(hits.sum().item())
            total_length += int(env.episode_steps[done_indices].sum().item())
            completed += int(done_indices.numel())
            env.reset(dones)
            states = env.state()
        else:
            states = next_states

    policy.train()
    return {
        "win_rate": wins / episodes,
        "return_rate": returns / episodes,
        "avg_agent_hits": total_hits / episodes,
        "avg_episode_length": total_length / episodes,
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    device = choose_device(args.device)
    set_seed(args.seed)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    num_envs = args.num_envs
    if num_envs is None:
        num_envs = 4096 if device.type == "cuda" else 512

    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 2048 if device.type == "cuda" else 512

    print(
        f"device={device} num_envs={num_envs} batch_size={batch_size} "
        f"total_transitions={args.total_transitions:,}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)} torch={torch.__version__}")

    policy = DuelingDQN(
        VectorizedPong.STATE_DIM,
        VectorizedPong.ACTION_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    target = DuelingDQN(
        VectorizedPong.STATE_DIM,
        VectorizedPong.ACTION_DIM,
        hidden_dim=args.hidden_dim,
    ).to(device)
    target.load_state_dict(policy.state_dict())
    target.eval()

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    replay = DeviceReplayBuffer(
        capacity=args.replay_capacity,
        state_dim=VectorizedPong.STATE_DIM,
        device=device,
    )
    env = VectorizedPong(
        num_envs=num_envs,
        device=device,
        seed=args.seed,
        config=PongConfig(opponent_speed=args.curriculum_start_speed),
        reward_shaping=True,
    )
    states = env.state()

    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    print(f"policy_parameters={parameter_count:,}")

    vector_steps = math.ceil(args.total_transitions / num_envs)
    transitions = 0
    learner_updates = 0
    episode_counter = torch.zeros((), device=device, dtype=torch.long)
    win_counter = torch.zeros((), device=device, dtype=torch.long)
    return_counter = torch.zeros((), device=device, dtype=torch.long)
    next_eval = args.eval_every
    started = time.perf_counter()

    for vector_step in range(1, vector_steps + 1):
        progress = transitions / max(args.total_transitions, 1)
        curriculum_fraction = min(progress / max(args.curriculum_fraction, 1e-8), 1.0)
        opponent_speed = args.curriculum_start_speed + curriculum_fraction * (
            args.curriculum_end_speed - args.curriculum_start_speed
        )
        env.set_opponent_speed(opponent_speed)

        epsilon = epsilon_at(
            transitions,
            args.epsilon_start,
            args.epsilon_end,
            args.epsilon_decay_transitions,
        )
        with torch.no_grad():
            greedy_actions = policy(states).argmax(dim=1)
            random_actions = torch.randint(
                0, VectorizedPong.ACTION_DIM, (num_envs,), device=device
            )
            explore = torch.rand(num_envs, device=device) < epsilon
            actions = torch.where(explore, random_actions, greedy_actions)

        next_states, rewards, dones, wins, _ = env.step(actions)
        replay.add_batch(states, actions, rewards, next_states, dones)

        transitions += num_envs
        episode_counter += dones.sum()
        win_counter += wins.sum()
        return_counter += ((env.agent_hits > 0) & dones).sum()
        env.reset(dones)
        states = env.state()

        if (
            len(replay) >= max(args.warmup_transitions, batch_size)
            and vector_step % args.train_every == 0
        ):
            for _ in range(args.gradient_steps):
                batch = replay.sample(batch_size)
                batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch

                current_q = policy(batch_states).gather(
                    1, batch_actions.unsqueeze(1)
                ).squeeze(1)
                with torch.no_grad():
                    next_actions = policy(batch_next_states).argmax(dim=1, keepdim=True)
                    next_q = target(batch_next_states).gather(1, next_actions).squeeze(1)
                    targets = batch_rewards + args.gamma * next_q * (1.0 - batch_dones)

                loss = nn.functional.smooth_l1_loss(current_q, targets)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                learner_updates += 1

                if learner_updates % args.target_update_every == 0:
                    target.load_state_dict(policy.state_dict())

        crossed_eval = transitions >= next_eval or vector_step == vector_steps
        if crossed_eval:
            elapsed = max(time.perf_counter() - started, 1e-9)
            eval_metrics = evaluate(
                policy=policy,
                device=device,
                episodes=args.eval_episodes,
                num_envs=min(num_envs, args.eval_envs),
                seed=args.seed + transitions,
                opponent_speed=args.curriculum_end_speed,
            )
            episodes_completed = int(episode_counter.item())
            training_wins = int(win_counter.item())
            training_returns = int(return_counter.item())
            train_win_rate = training_wins / max(episodes_completed, 1)
            train_return_rate = training_returns / max(episodes_completed, 1)
            print(
                f"transitions={transitions:,} updates={learner_updates:,} "
                f"epsilon={epsilon:.3f} opponent_speed={opponent_speed:.2f} "
                f"throughput={transitions / elapsed:,.0f} transitions/s "
                f"train_win={train_win_rate:.3f} train_return={train_return_rate:.3f} "
                f"eval_win={eval_metrics['win_rate']:.3f} "
                f"eval_return={eval_metrics['return_rate']:.3f} "
                f"eval_hits={eval_metrics['avg_agent_hits']:.2f}"
            )
            next_eval += args.eval_every

    elapsed = time.perf_counter() - started
    final_metrics = evaluate(
        policy=policy,
        device=device,
        episodes=args.eval_episodes,
        num_envs=min(num_envs, args.eval_envs),
        seed=args.seed + 10_000_000,
        opponent_speed=args.curriculum_end_speed,
    )

    checkpoint = {
        "model_state_dict": policy.state_dict(),
        "state_dim": VectorizedPong.STATE_DIM,
        "action_dim": VectorizedPong.ACTION_DIM,
        "hidden_dim": args.hidden_dim,
        "pong_config": asdict(PongConfig(opponent_speed=args.curriculum_end_speed)),
        "training_args": vars(args),
        "final_metrics": final_metrics,
        "parameter_count": parameter_count,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    summary = {
        **final_metrics,
        "device": str(device),
        "num_envs": num_envs,
        "batch_size": batch_size,
        "parameter_count": parameter_count,
        "transitions": transitions,
        "learner_updates": learner_updates,
        "elapsed_seconds": elapsed,
        "transitions_per_second": transitions / max(elapsed, 1e-9),
        "checkpoint": str(output_path),
    }
    print(json.dumps(summary, indent=2))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a vectorized Double/Dueling DQN Pong agent."
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")
    parser.add_argument("--total-transitions", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--warmup-transitions", type=int, default=50_000)
    parser.add_argument("--train-every", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--target-update-every", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-transitions", type=int, default=1_000_000)
    parser.add_argument("--curriculum-start-speed", type=float, default=2.5)
    parser.add_argument("--curriculum-end-speed", type=float, default=5.0)
    parser.add_argument("--curriculum-fraction", type=float, default=0.70)
    parser.add_argument("--eval-every", type=int, default=250_000)
    parser.add_argument("--eval-episodes", type=int, default=1_000)
    parser.add_argument("--eval-envs", type=int, default=1_024)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="pong_vectorized.pth")
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
