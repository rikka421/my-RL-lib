import time

import gymnasium as gym
import torch
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt

from my_rl_library.agents.DQNs.MyDQN import MyDQN


def train(env_name, Agent, myModel, plot_name="DQN", timestep=1e5, double_q=True, dueling_q=False, priority_pool=True):
    # 创建 Gym 环境并添加 Monitor 以记录数据
    env = gym.make(env_name)

    env = Monitor(env, log_dir + "monitor.csv")

    time1 = time.time()

    if myModel:
        # 初始化模型，启用 TensorBoard 日志记录
        model = Agent("MlpPolicy", env, double_q=double_q, dueling_q=dueling_q, priority_pool=priority_pool)
        # 开始训练
        model.train(total_timesteps=timestep)
    else:
        model = Agent("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)
        # 开始训练
        model.learn(total_timesteps=timestep)


    time2 = time.time()

    print(time2-time1)

    model.save(log_dir + "model_weights")
    env.close()

    if myModel:
        plot(env_name, plot_name)
    else:
        plot(env_name, "base-DQN")


def plot(env_name, agent_name):
    # 读取 Monitor 日志文件
    monitor_data = pd.read_csv(log_dir + "monitor.csv", skiprows=1)

    # 获取奖励数据
    rewards = monitor_data["r"]

    # 定义滑动窗口大小
    window_size = 10

    # 计算滑动平均值
    smoothed_rewards = rewards.rolling(window=window_size).mean()

    # 绘制平滑后的奖励曲线
    plt.plot(smoothed_rewards, label=agent_name + " " + env_name)

def run_tests():
    discrete_envs_list = ["CartPole-v1"]
    timestep = 2e5

    for env_name in discrete_envs_list:
        train(env_name, MyDQN, plot_name="DQN", myModel=True, timestep=timestep, double_q=False, dueling_q=False, priority_pool=False)
        train(env_name, MyDQN, plot_name="priority-pool", myModel=True, timestep=timestep, double_q=False, dueling_q=False, priority_pool=True)
        train(env_name, MyDQN, plot_name="double_q", myModel=True, timestep=timestep, double_q=True, dueling_q=False, priority_pool=False)
        train(env_name, MyDQN, plot_name="priority-pool+double_q", myModel=True, timestep=timestep, double_q=True, dueling_q=False, priority_pool=True)
        train(env_name, MyDQN, plot_name="dueling+DQN", myModel=True, timestep=timestep, double_q=False, dueling_q=True, priority_pool=False)
        train(env_name, MyDQN, plot_name="dueling+priority-pool", myModel=True, timestep=timestep, double_q=True, dueling_q=False, priority_pool=True)
        train(env_name, MyDQN, plot_name="dueling+double_q", myModel=True, timestep=timestep, double_q=True, dueling_q=True, priority_pool=False)
        train(env_name, MyDQN, plot_name="dueling+priority-pool+double_q", myModel=True, timestep=timestep, double_q=True, dueling_q=True, priority_pool=True)


    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Smoothed Training Rewards over Episodes")
    plt.legend()
    plt.show()



import cProfile
import pstats
if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)

    # 设置日志保存路径
    log_dir = "./logs/"
    tensorboard_log = "./agent_cartpole_tensorboard/"
    if os.path.exists(tensorboard_log):
        # 清空 log_dir 目录中的所有内容
        shutil.rmtree(tensorboard_log)
    # 创建空的 log_dir 目录
    os.makedirs(log_dir, exist_ok=True)

    run_tests()

else:
    # 使用 cProfile 分析 my_function 的性能
    cProfile.run('run_tests()', 'output.prof')

    # 显示结果
    with open('result.txt', 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats('time').print_stats()


"""
    49937   28.157    0.001   30.112    0.001 /home/rikka/files/codes/my-RL-lib/my_rl_library/utils/SamplePool.py:96(sample)
    62412   23.009    0.000   23.009    0.000 {method 'run_backward' of 'torch._C._EngineBase' objects}
    49937   11.037    0.000   16.997    0.000 /home/rikka/files/codes/my-RL-lib/my_rl_library/agents/DQNs/MyDQN.py:23(update_samples)
   796569   10.956    0.000   10.956    0.000 {built-in method torch._C._nn.linear}
    49937    5.801    0.000   16.291    0.000 /home/rikka/anaconda3/envs/RL/lib/python3.10/site-packages/torch/optim/adam.py:320(_single_tensor_adam)
   420014    4.251    0.000    4.251    0.000 {built-in method torch.tensor}
    49937    3.385    0.000  122.618    0.002 /home/rikka/files/codes/my-RL-lib/my_rl_library/agents/DQNs/MyDQN.py:62(step)



    49937   32.243    0.001   34.467    0.001 /home/rikka/files/codes/my-RL-lib/my_rl_library/utils/SamplePool.py:96(sample)
    62412   28.033    0.000   28.033    0.000 {method 'run_backward' of 'torch._C._EngineBase' objects}
    49937   14.457    0.000   20.974    0.000 /home/rikka/files/codes/my-RL-lib/my_rl_library/agents/DQNs/MyDQN.py:23(update_samples)
   797316   13.953    0.000   13.953    0.000 {built-in method torch._C._nn.linear}
    49937    7.487    0.000   21.473    0.000 /home/rikka/anaconda3/envs/RL/lib/python3.10/site-packages/torch/optim/adam.py:320(_single_tensor_adam)
   420263    6.313    0.000    6.313    0.000 {built-in method torch.tensor}
    49937    4.538    0.000  151.403    0.003 /home/rikka/files/codes/my-RL-lib/my_rl_library/agents/DQNs/MyDQN.py:63(step)
"""