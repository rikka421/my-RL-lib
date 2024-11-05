class Agent():
    def __init__(self):
        pass

    def take_action(self, state):
        raise NotImplementedError

    def update(self, data):
        # 根据方法的不同, 传入的data也不同.
        raise NotImplementedError
