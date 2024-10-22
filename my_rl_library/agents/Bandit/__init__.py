from my_rl_library.agents.Bandit.BernoulliBanditSolver import Solver
from my_rl_library.agents.Bandit.EpsilonGreedySolver import EpsilonGreedy
from my_rl_library.agents.Bandit.DecayingEpsilonGreedy import DecayingEpsilonGreedy

import matplotlib.pyplot as plt

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

