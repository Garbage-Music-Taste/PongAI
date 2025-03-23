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

score1 = 0  # Bottom player (paddle1)
score2 = 0  # Top player (paddle2)

font = pygame.font.SysFont(None, 36)


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

    #if keys[pygame.K_a]:
     #   paddle2.update(-1)
    #if keys[pygame.K_d]:
    #    paddle2.update(1)

    if ball.position[0] < paddle2.x + paddle2.length // 2:
        paddle2.update(-1)
    elif ball.position[0] > paddle2.x + paddle2.length // 2:
        paddle2.update(1)

    ball.update()

    if ball.position[0] <= 0 or ball.position[0] >= WIDTH:
        ball.bounce_x()

    # Scoring
    if ball.position[1] <= 0:  # Player 1 scores
        score1 += 1
        ball = Ball([WIDTH // 2, HEIGHT // 2], [4, 4])
    elif ball.position[1] >= HEIGHT:  # Player 2 scores
        score2 += 1
        ball = Ball([WIDTH // 2, HEIGHT // 2], [4, -4])

    if ball.get_rect().colliderect(paddle1.get_rect()) and ball.velocity[1] > 0:
        ball.paddle_bounce(paddle1)
    if ball.get_rect().colliderect(paddle2.get_rect()) and ball.velocity[1] < 0:
        ball.paddle_bounce(paddle2)

    pygame.draw.rect(screen, WHITE, paddle1.get_rect())
    pygame.draw.rect(screen, WHITE, paddle2.get_rect())
    pygame.draw.circle(screen, WHITE, (int(ball.position[0]), int(ball.position[1])), ball.radius)

    for x in range(0, WIDTH, 20):
        if (x // 20) % 2 == 0:
            pygame.draw.line(screen, WHITE, (x, HEIGHT // 2), (x + 10, HEIGHT // 2))

    score_text = font.render(f"{score2} - {score1}", True, WHITE)
    screen.blit(score_text, (WIDTH // 2 - 30, HEIGHT // 2 + 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
