import pygame
import sys

from Ball import Ball
from Paddle import Paddle

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

paddle1 = Paddle([WIDTH // 2 - 50, HEIGHT - 30], 100)
paddle2 = Paddle([WIDTH // 2 - 50, 10], 100)  # top paddle


ball = Ball([WIDTH // 2, HEIGHT // 2], [4, -4])

running = True
while running:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        paddle1.update(-1)
    if keys[pygame.K_RIGHT]:
        paddle1.update(1)
    if keys[pygame.K_a]:
        paddle2.update(-1)
    if keys[pygame.K_d]:
        paddle2.update(1)

    ball.update()

    if ball.position[0] <= 0 or ball.position[0] >= WIDTH:
        ball.bounce_x()
    if ball.position[1] <= 0 or ball.position[1] >= HEIGHT:
        print("Someone scored lol")
        ball.bounce_y()

    if ball.get_rect().colliderect(paddle1.get_rect()) and ball.velocity[1] > 0:
        ball.bounce_y()
    if ball.get_rect().colliderect(paddle2.get_rect()) and ball.velocity[1] < 0:
        ball.bounce_y()

    pygame.draw.rect(screen, WHITE, paddle1.get_rect())
    pygame.draw.rect(screen, WHITE, paddle2.get_rect())
    pygame.draw.circle(screen, WHITE, (int(ball.position[0]), int(ball.position[1])), ball.radius)

    for x in range(0, WIDTH, 20):
        if (x // 20) % 2 == 0:
            pygame.draw.line(screen, WHITE, (x, HEIGHT // 2), (x + 10, HEIGHT // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
