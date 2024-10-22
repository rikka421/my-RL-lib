import numpy as np
import matplotlib.pyplot as plt

from my_rl_library.agents.Bandit.BernoulliBanditSolver import Solver

class DecayingEpsilonGreedy(Solver):
    """ epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类 """
    def __init__(self, bandit, init_prob=1.0, name="DecayingEpsilon"):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.estimates = np.array([init_prob] * self.bandit.K)
        self.total_count = 0
        self.__name__ = name

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


if __name__ == '__main__':
    def plot_results(solvers, solver_names):
        """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
        而solver_names也是一个列表,存储每个策略的名称"""
        for idx, solver in enumerate(solvers):
            time_list = range(len(solver.regrets))
            plt.plot(time_list, solver.regrets, label=solver_names[idx])
        plt.xlabel('Time steps')
        plt.ylabel('Cumulative regrets')
        plt.title('%d-armed bandit' % solvers[0].bandit.K)
        plt.legend()
        plt.show()


    from my_rl_library.envs.BanditEnv.BernoulliBandit import BernoulliBandit
    K = 10
    bandit = BernoulliBandit(K)
    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit)
    decaying_epsilon_greedy_solver.run(5000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results([decaying_epsilon_greedy_solver], ["DecayingEpsilonGreedy"])

    # epsilon值衰减的贪婪算法的累积懊悔为：10.114334931260183