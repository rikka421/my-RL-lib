from my_rl_library.envs.MDP import MDP
import numpy as np


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

        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.transition_probs = self.init_transition_probs()
        super().__init__(self.states_num, self.actions_num, self.transition_probs, self.rewards,
                         self.start_state, self.terminal_states)

    def get_reward(self, state, action=None):
        return self.rewards[state]

    def init_transition_probs(self):
        self.transition_probs = np.zeros((self.states_num, self.actions_num, self.states_num), dtype=float)

        direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for i in range(self.states_num):
            if i in self.terminal_states:
                continue
            for j in range(self.actions_num):
                a = direction[j]
                s = (i // self.row, i % self.col)
                ns = (s[0] + a[0], s[1] + a[1])
                if 0 < ns[0] <= self.row and 0 < ns[1] <= self.col:
                    pass
                else:
                    ns = s
                ni = ns[0] + ns[1]
                self.transition_probs[i, j, ni] = 1

        for s in self.terminal_states:
            self.transition_probs[s, :, s] = np.ones(self.actions_num)

        return self.transition_probs


if __name__ == "__main__":
    # 模拟一次随机游走
    mdp = CliffWalkingEnv(3, 3)
    print(mdp.terminal_states)
    print(mdp.rewards)
    print(mdp.transition_probs)

    s = mdp.reset()
    done = True
    while not done:
        action = np.random.choice([0, 1, 2, 3])
        ns, r, done = mdp.step(s, action)
        print(s, action, ns, r, done)
        s = ns