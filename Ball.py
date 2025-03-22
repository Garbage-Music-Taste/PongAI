from typing import Tuple


class Ball:
    def __int__(self, initial_position: Tuple[float, float] = (1.0, 2.0)):
        self.position = initial_position
        self.velocity = [0,0]

    def update(self):
        self.position[0] += self.velocity


