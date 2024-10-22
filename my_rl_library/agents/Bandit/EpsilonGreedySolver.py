import numpy as np
import matplotlib.pyplot as plt

from my_rl_library.agents.Bandit.BernoulliBanditSolver import Solver

class EpsilonGreedy(Solver):
    """ epsilon贪婪算法,继承Solver类 """
    def __init__(self, bandit, epsilon=0.01, init_prob=1.0, init_N=0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        #初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.K)

        self.sum_count = 0

        self.init_N = init_N

    def run_one_step(self):
        # 如果有一个固定的init_N, 就挨个尝试; 否则, 按照epsilon-greedy尝试
        if self.sum_count < self.init_N:
            k = self.sum_count % self.bandit.K
        else:
            if np.random.random() < self.epsilon:
                k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
            else:
                k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r -
                                                          self.estimates[k])

        self.sum_count += 1
        return k