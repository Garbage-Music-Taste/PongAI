import pygame
import sys
import torch
import numpy as np
from Ball import Ball
from Paddle import Paddle
from RL.Agent import Agent
from PongEnv import PongEnv

pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Create and load AI agent
agent = Agent(PongEnv.STATE_DIM, PongEnv.ACTION_DIM)
agent.policy_net.load_state_dict(torch.load("trainer.pth"))
agent.policy_net.eval()  # Set to evaluation mode
agent.epsilon = 0.0  # Ensure deterministic policy (no exploration)

# Initialize game elements
paddle1 = Paddle([WIDTH // 2 - 50, HEIGHT - 30], 100)  # AI player (bottom)
paddle2 = Paddle([WIDTH // 2 - 50, 10], 100)  # Human player (top)
ball = Ball([WIDTH // 2, HEIGHT // 2], [0,-7])  # Initial speed set to 7
score1 = 0  # AI player score
score2 = 0  # Human player score
font = pygame.font.SysFont(None, 36)


# Function to get the game state in the format expected by the AI
def get_state():
    return np.array([
        ball.position[0] / WIDTH,  # ball x
        ball.position[1] / HEIGHT,  # ball y
        ball.velocity[0] / Ball.max_speed,  # vx
        ball.velocity[1] / Ball.max_speed,  # vy
        paddle1.x / WIDTH,  # AI paddle x (bottom)
        paddle2.x / WIDTH,  # human paddle x (top)
        (paddle1.x + paddle1.length / 2 - ball.position[0]) / WIDTH  # dist to ball
    ], dtype=np.float32)


running = True
while running:
    screen.fill(BLACK)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Human player controls (top paddle)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        paddle2.update(-1)
    if keys[pygame.K_RIGHT]:
        paddle2.update(1)

    # AI player action (bottom paddle)
    state = get_state()
    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor)
        ai_action = torch.argmax(q_values).item()

    # AI movement (0 = left, 1 = stay, 2 = right)
    if ai_action == 0:
        paddle1.update(-1)
    elif ai_action == 2:
        paddle1.update(1)

    # Update ball
    ball.update()

    # Wall bounce
    if ball.position[0] <= 0 or ball.position[0] >= WIDTH:
        ball.bounce_x()

    # Scoring
    if ball.position[1] <= 0:  # AI scores
        score1 += 1
        ball = Ball([WIDTH // 2, HEIGHT // 2], [0,7])  # Reset with speed 8
    elif ball.position[1] >= HEIGHT:  # Human scores
        score2 += 1
        ball = Ball([WIDTH // 2, HEIGHT // 2], [0,7])  # Reset with speed 8

    # Paddle collisions
    if ball.get_rect().colliderect(paddle1.get_rect()) and ball.velocity[1] > 0:
        ball.paddle_bounce(paddle1)
    if ball.get_rect().colliderect(paddle2.get_rect()) and ball.velocity[1] < 0:
        ball.paddle_bounce(paddle2)

    # Draw everything
    pygame.draw.rect(screen, WHITE, paddle1.get_rect())
    pygame.draw.rect(screen, WHITE, paddle2.get_rect())
    pygame.draw.circle(screen, WHITE, (int(ball.position[0]), int(ball.position[1])), ball.radius)

    # Draw center line
    for x in range(0, WIDTH, 20):
        if (x // 20) % 2 == 0:
            pygame.draw.line(screen, WHITE, (x, HEIGHT // 2), (x + 10, HEIGHT // 2))

    # Draw score
    score_text = font.render(f"You: {score2}  AI: {score1}", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - 60, HEIGHT // 2 + 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()