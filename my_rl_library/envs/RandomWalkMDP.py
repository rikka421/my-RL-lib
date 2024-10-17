from my_rl_library.envs.MDP import MDP
import numpy as np


class RandomWalkMDP(MDP):
    def __init__(self, n):
        assert n >= 2
        # 定义状态、动作和奖励
        self.n = n
        self.states = list(range(n))
        self.actions = ['left', 'right']
        self.terminal_states = [0, n-1]
        self.rewards = {n-1: 1}
        self.transition_probs = {}
        super().__init__(self.states, self.actions, self.transition_probs, self.rewards)

    def init_transition_probs(self):
        for i in range(1, self.n-1):
            self.transition_probs[i] = {}
        self.transition_probs[1] = {"left": {0: 1.0}}
        self.transition_probs[self.n-2] = {"right": {self.n-1: 1.0}}
        for l in range(1, self.n-2):
            r = l + 1
            self.transition_probs[l]["right"] = {r: 1.0}
            self.transition_probs[r]["left"] = {l: 1.0}


    def get_reward(self, state, action=None):
        return self.rewards.get(state, 0)


if __name__ == "__main__":
    # 创建 MDP 实例
    mdp = RandomWalkMDP(8)

    # 模拟一次随机游走
    current_state = 2  # 从状态 2 开始
    done = False
    while not done:
        action = np.random.choice(mdp.actions)  # 随机选择动作
        next_state, reward, done = mdp.step(current_state, action)
        print(f"Current State: {current_state}, Action: {action}, Next State: {next_state}, Reward: {reward}")
        current_state = next_state
