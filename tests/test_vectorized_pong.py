import torch

from VectorizedPong import PongConfig, VectorizedPong
from RL.VectorDQN import DeviceReplayBuffer, DuelingDQN


def test_shapes_and_finite_states():
    env = VectorizedPong(32, "cpu", seed=1)
    states = env.state()
    assert states.shape == (32, 7)
    assert torch.isfinite(states).all()

    actions = torch.randint(0, 3, (32,))
    next_states, rewards, dones, wins, hits = env.step(actions)
    assert next_states.shape == (32, 7)
    assert rewards.shape == dones.shape == wins.shape == hits.shape == (32,)
    assert torch.isfinite(next_states).all()
    assert torch.isfinite(rewards).all()


def test_paddles_stay_in_bounds():
    config = PongConfig()
    env = VectorizedPong(16, "cpu", seed=2, config=config)
    for _ in range(100):
        env.step(torch.zeros(16, dtype=torch.long))
    assert torch.all(env.paddle1_x >= 0)
    assert torch.all(env.paddle1_x <= config.width - config.paddle_length)

    for _ in range(100):
        env.step(torch.full((16,), 2, dtype=torch.long))
    assert torch.all(env.paddle1_x >= 0)
    assert torch.all(env.paddle1_x <= config.width - config.paddle_length)


def test_replay_and_network_train_step():
    device = torch.device("cpu")
    env = VectorizedPong(64, device, seed=3)
    states = env.state()
    actions = torch.randint(0, 3, (64,), device=device)
    next_states, rewards, dones, _, _ = env.step(actions)

    replay = DeviceReplayBuffer(256, 7, device)
    replay.add_batch(states, actions, rewards, next_states, dones)
    batch = replay.sample(32)

    policy = DuelingDQN(7, 3, hidden_dim=32)
    target = DuelingDQN(7, 3, hidden_dim=32)
    target.load_state_dict(policy.state_dict())
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
    q_values = policy(batch_states).gather(1, batch_actions[:, None]).squeeze(1)
    with torch.no_grad():
        next_actions = policy(batch_next_states).argmax(1, keepdim=True)
        next_q = target(batch_next_states).gather(1, next_actions).squeeze(1)
        targets = batch_rewards + 0.97 * next_q * (1.0 - batch_dones)

    loss = torch.nn.functional.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
