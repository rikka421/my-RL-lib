
class MyEnv():
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError