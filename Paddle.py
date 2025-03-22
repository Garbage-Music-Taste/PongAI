class Paddle:
    
    def __init__(self, initial_position, length):
        self.x, self.y = initial_position
        self.length = length

    def update_position(self, direction, step):
        self.y += direction * step
    
    def get_posiiton(self):
        return (self.x, self.y)
    
    def set_position(self, initial_position):
        self.x, self.y = initial_position