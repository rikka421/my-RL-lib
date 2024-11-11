import torch

class Agent():
    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.target_network = None

    def predict(self, state, deterministic=True):
        self.target_network.eval()

    def learn(self, total_timesteps=1e5):
        # 根据方法的不同, 传入的data也不同.
        raise NotImplementedError

    def save(self, path):
        torch.save(self.target_network.state_dict(), path + self.name + '.pth')

    def load(self, path):
        self.target_network.load_state_dict(torch.load(path + self.name + '.pth'))

