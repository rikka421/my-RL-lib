import torch

class Agent():
    def __init__(self, name, env):
        self.name = name
        self.env = env
        self.tar_model = None

    def predict(self, state, deterministic=True):
        self.tar_model.eval()

    def learn(self, total_timesteps=1e5):
        # 根据方法的不同, 传入的data也不同.
        raise NotImplementedError

    def save(self, path):
        torch.save(self.tar_model.state_dict(), path + self.name + '.pth')

    def load(self, path):
        self.tar_model.load_state_dict(torch.load(path + self.name + '.pth'))

