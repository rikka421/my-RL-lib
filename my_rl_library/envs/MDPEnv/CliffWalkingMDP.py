import numpy as np
from my_rl_library import MDP


class CliffWalkingEnv(MDP):
    """ 悬崖漫步环境"""
    def __init__(self, row=4, col=12):
        self.row = row  # 定义网格世界的行
        self.col = col  # 定义网格世界的列
        self.states_num = self.col * self.row
        self.actions_num = 4

        self.start_state = 0
        self.terminal_states = list(range(1, self.col))
        self.rewards = [0] * self.states_num
        for s in self.terminal_states:
            self.rewards[s] = -100
        self.rewards[self.col - 1] = 10
        self.rewards = np.array(self.rewards)
        self.rewards = self.rewards.reshape((-1, 1))
        self.rewards = np.tile(self.rewards, (1, self.actions_num))

        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.transition_probs = self.init_transition_probs()
        super().__init__(self.states_num, self.actions_num, self.transition_probs, self.rewards,
                         self.start_state, self.terminal_states)


    def init_transition_probs(self):
        self.transition_probs = np.zeros((self.states_num, self.actions_num, self.states_num), dtype=float)

        direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i in range(self.states_num):
            if i in self.terminal_states:
                continue
            for j in range(self.actions_num):
                a = direction[j]
                xy = (i // self.col, i % self.col)
                nxy = (xy[0] + a[0], xy[1] + a[1])
                ni = nxy[0] * self.col + nxy[1] if (0 <= nxy[0] < self.row and 0 <= nxy[1] < self.col) else i
                self.transition_probs[i, j, ni] = 1

        for s in self.terminal_states:
            self.transition_probs[s, :, s] = np.zeros(self.actions_num)

        return self.transition_probs


if __name__ == "__main__":
    # 模拟一次随机游走
    mdp = CliffWalkingEnv(5, 10)

    s = mdp.reset()
    done = False
    while not done:
        # action = np.random.choice([0, 1, 2, 3])
        action = int(input())

        ns, r, done = mdp.step(s, action)
        print(s, action, ns, r, done)
        s = ns