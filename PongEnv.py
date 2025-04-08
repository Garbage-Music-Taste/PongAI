import math
import random

import pygame
import numpy as np
from Ball import Ball
from Paddle import Paddle

WIDTH, HEIGHT = 800, 600
MAX_SCORE = 10


class PongEnv:
    STATE_DIM = 7
    ACTION_DIM = 3

    def __init__(self, render_mode=False, episode_num=None):
        pygame.init()
        self.render_mode = render_mode
        self.episode_num = episode_num
        if render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
            pygame.font.init()
            self.font = pygame.font.SysFont(None, 36)

        self.reset()


        self.score1 = 0
        self.score2 = 0

    def reset(self):
        self.paddle1 = Paddle([WIDTH // 2 - 50, HEIGHT - 30], 100)
        self.paddle2 = Paddle([WIDTH // 2 - 50, 10], 100)
        if random.random() < 1:
            angle = (2*random.randint(0,1) - 1) * random.uniform(np.pi/2 + np.pi/6, np.pi - np.pi/6)
        else:
            angle = (2*random.randint(0,1) - 1) * random.uniform(np.pi/6, np.pi/2 - np.pi/6)
        self.ball = Ball([WIDTH // 2, HEIGHT // 2], [7 * np.cos(angle), 7 * np.sin(angle)])

        self.score1 = 0
        self.score2 = 0
        return self.get_state()

    def step(self, action):
        """
        Action: 0 = left, 1 = stay, 2 = right (for paddle1/agent)
        Paddle2 can be controlled by simple AI or fixed

        Returns state, reward, done
        """
        # --- Paddle 1 (agent) ---
        if action == 0:
            self.paddle1.update(-1)
        elif action == 2:
            self.paddle1.update(1)

        # --- Paddle 2 (simple AI, follow the ball) ---
        if self.ball.position[0] < self.paddle2.x + self.paddle2.length // 2:
            self.paddle2.update(-1)
        elif self.ball.position[0] > self.paddle2.x + self.paddle2.length // 2:
            self.paddle2.update(1)

        self.ball.update()

        # --- Wall bounce ---
        if self.ball.position[0] <= 0 or self.ball.position[0] >= WIDTH:
            self.ball.bounce_x()

        reward = 0
        done = False

        # --- Scoring ---
        if self.ball.position[1] <= 0:
            self.score1 += 1
            reward += 1
            win = 1
            done = True
        elif self.ball.position[1] >= HEIGHT:
            self.score2 += 1
            paddle_center = self.paddle1.x + self.paddle1.length / 2
            ball_x = self.ball.position[0]
            distance = abs(paddle_center - ball_x)
            max_distance = WIDTH / 2

            # Quadratic penalty, more forgiving when close
            penalty = -1

            reward += penalty
            done = True

        # --- Paddle collisions ---
        if self.ball.get_rect().colliderect(self.paddle1.get_rect()) and self.ball.velocity[1] > 0:
            self.ball.paddle_bounce(self.paddle1)

        if self.ball.get_rect().colliderect(self.paddle2.get_rect()) and self.ball.velocity[1] < 0:
            self.ball.paddle_bounce(self.paddle2)

        if self.render_mode:
            self.render()

        reward = 0

        # Base reward for keeping the ball in play
        reward += 0.01  # Small positive reward per frame

        # Encourage the paddle to stay under the ball
        paddle_center = self.paddle1.x + self.paddle1.length / 2
        ball_x = self.ball.position[0]
        distance = abs(paddle_center - ball_x)
        max_distance = WIDTH / 2

        # Distance-based reward: higher when paddle is aligned with ball
        reward += 0.05 * (1 - min(distance, max_distance) / max_distance)


        return self.get_state(), reward, done

    def get_state(self):
        # dimension 7
        v = np.array([
            self.ball.position[0] / WIDTH,  # ball x
            self.ball.position[1] / HEIGHT,  # ball y
            self.ball.velocity[0] / Ball.max_speed,  # vx
            self.ball.velocity[1] / Ball.max_speed,  # vy
            self.paddle1.x / WIDTH,  # agent paddle x
            self.paddle2.x / WIDTH,  # opponent paddle x ✅
            (self.paddle1.x + self.paddle1.length / 2 - self.ball.position[0]) / WIDTH  # dist to ball ✅
        ], dtype=np.float32)
        assert len(v) == PongEnv.STATE_DIM
        return v

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255), self.paddle1.get_rect())
        pygame.draw.rect(self.screen, (255, 255, 255), self.paddle2.get_rect())
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.ball.position[0]), int(self.ball.position[1])),
                           self.ball.radius)

        # Draw horizontal dotted line
        for x in range(0, WIDTH, 20):
            if (x // 20) % 2 == 0:
                pygame.draw.line(self.screen, (255, 255, 255), (x, HEIGHT // 2), (x + 10, HEIGHT // 2))

        if self.episode_num is not None:
            text = self.font.render(f"Episode {self.episode_num}", True, (255, 255, 255))
            text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40))
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(60)
