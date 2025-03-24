import pygame


class Ball:
    max_speed = 4

    def __init__(self, initial_position: list[float], velocity: list[float]):
        self.position = initial_position
        self.velocity = velocity
        self.radius = 10
        self.speed_increment = 1.05  # pong mechanics, speed up the ball (more fun ig)

    def update(self):
        self.position[0] += self.velocity[0]
        self.position[1] += self.velocity[1]

    def bounce_x(self):
        self.velocity[0] *= -1

    def bounce_y(self):
        self.velocity[1] *= -1

    def get_rect(self):
        return pygame.Rect(
            self.position[0] - self.radius,
            self.position[1] - self.radius,
            self.radius * 2,
            self.radius * 2
        )

    def paddle_bounce(self, paddle):
        self.bounce_y()

        paddle_center = paddle.x + paddle.length / 2
        dist = (self.position[0] - paddle_center) / (paddle.length / 2)
        self.velocity[0] += dist * 2  # tweak multiplier as needed

        speed = (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
        if speed > self.max_speed:
            factor = self.max_speed / speed
            self.velocity[0] *= factor
            self.velocity[1] *= factor
        else:
            self.velocity[0] *= self.speed_increment
            self.velocity[1] *= self.speed_increment
