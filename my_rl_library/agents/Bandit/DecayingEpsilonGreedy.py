import numpy as np
import matplotlib.pyplot as plt

from my_rl_library.agents.Bandit.BernoulliBanditSolver import Solver

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0

    def run_one_step(self):
        self.total_count += 1
        if np.random.random() < 1 / self.total_count:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.K)
        else:
            k = np.argmax(self.estimates)

        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (r -
                                                          self.estimates[k])

        return k


        np.random.seed(1)
        decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
        decaying_epsilon_greedy_solver.run(5000)
        print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
        plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

        # epsilon值衰减的贪婪算法的累积懊悔为：10.114334931260183