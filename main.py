import pygame
import sys

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

paddle_width, paddle_height = 100, 20
paddle_x = (WIDTH - paddle_width) // 2
paddle_y = HEIGHT - 40
paddle_rect = pygame.Rect(paddle_x, paddle_y, paddle_width, paddle_height)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    pygame.draw.rect(screen, WHITE, paddle_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()
