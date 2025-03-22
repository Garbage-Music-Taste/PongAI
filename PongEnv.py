import pygame
import numpy as np
from Ball import Ball
from Paddle import Paddle

WIDTH, HEIGHT = 800, 600
MAX_SCORE = 10


class PongEnv:
    def __init__(self, render_mode=False):
        pygame.init()
        self.render_mode = render_mode
        if render_mode:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()

        self.paddle1 = Paddle([WIDTH // 2 - 50, HEIGHT - 30], 100)
        self.paddle2 = Paddle([WIDTH // 2 - 50, 10], 100)
        self.ball = Ball([WIDTH // 2, HEIGHT // 2], [4, -4])
        self.score1 = 0
        self.score2 = 0

    def reset(self):
        self.paddle1 = Paddle([WIDTH // 2 - 50, HEIGHT - 30], 100)
        self.paddle2 = Paddle([WIDTH // 2 - 50, 10], 100)
        self.ball = Ball([WIDTH // 2, HEIGHT // 2], [4, -4])
        self.score1 = 0
        self.score2 = 0
        return self.get_state()

    def step(self, action):
        """
        Action: 0 = left, 1 = stay, 2 = right (for paddle1/agent)
        Paddle2 can be controlled by simple AI or fixed
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
            reward = 1
            done = True
        elif self.ball.position[1] >= HEIGHT:
            self.score2 += 1
            reward = -1
            done = True

        # --- Paddle collisions ---
        if self.ball.get_rect().colliderect(self.paddle1.get_rect()) and self.ball.velocity[1] > 0:
            self.ball.paddle_bounce(self.paddle1)
        if self.ball.get_rect().colliderect(self.paddle2.get_rect()) and self.ball.velocity[1] < 0:
            self.ball.paddle_bounce(self.paddle2)

        if self.render_mode:
            self.render()

        return self.get_state(), reward, done

    def get_state(self):
        return np.array([
            self.ball.position[0] / WIDTH,
            self.ball.position[1] / HEIGHT,
            self.ball.velocity[0] / Ball.max_speed,  # Normalize assuming max speed
            self.ball.velocity[1] / Ball.max_speed,
            self.paddle1.x / WIDTH
        ], dtype=np.float32)

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

        pygame.display.flip()
        self.clock.tick(60)
