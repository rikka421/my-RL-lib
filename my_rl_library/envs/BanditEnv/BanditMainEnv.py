import numpy as np
import matplotlib.pyplot as plt

from my_rl_library.agents.Bandit import *

class BanditMainEnv():
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.results = []

    def run(self, num_steps):
        self.results = []
        for agent in self.agents:
            # 让 agent 在环境中运行并返回结果
            agent.run(num_steps)
            result = agent.regrets
            self.results.append(result)
            # print(f"Agent {agent.__name__} finished with result: {result}")
        return self.results

    def plot_results(self):
        """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
        而solver_names也是一个列表,存储每个策略的名称"""
        for idx, agent in enumerate(self.agents):
            result = agent.regrets
            time_list = range(len(result))
            plt.plot(time_list, result, label=agent.__name__)
        plt.xlabel('Time steps')
        plt.ylabel('Cumulative regrets')
        plt.title('%d-armed bandit' % self.env.K)
        plt.legend()
        plt.show()
