import pygame


class Ball:
    def __init__(self, initial_position: list[float], velocity: list[float]):
        self.position = initial_position
        self.velocity = velocity
        self.radius = 10

    def update(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def bounce_x(self):
        self.velocity[0] *= -1

    def bounce_y(self):
        self.velocity[1] *= -1

    def get_rect(self):
        return pygame.Rect(self.position[0] - self.radius,
                           self.position[1] - self.radius,
                           self.radius * 2,
                           self.radius * 2)
