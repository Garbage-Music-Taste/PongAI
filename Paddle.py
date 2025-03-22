import pygame


class Paddle:
    def __init__(self, initial_position: list[float], length):
        self.x, self.y = initial_position
        self.length = length
        self.height = 20
        self.speed = 5

    def update(self, direction, step=1):
        self.x += direction * self.speed * step

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.length, self.height)
