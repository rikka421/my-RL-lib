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


def train(env_name, Agent, myModel, timestep=1e5):
    # 创建 Gym 环境并添加 Monitor 以记录数据
    env = gym.make(env_name)

    env = Monitor(env, log_dir + "monitor.csv")

    time1 = time.time()

    if myModel:
        # 初始化模型，启用 TensorBoard 日志记录
        model = Agent("MlpPolicy", env, double_q=True, dueling_q=False, priority_pool=True)
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
        plot(env_name, "MyDQN-double-Q")
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

    for env_name in discrete_envs_list:
        train(env_name, DQN, myModel=False, timestep=1e4)
        train(env_name, MyDQN, myModel=True, timestep=1e4)

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Smoothed Training Rewards over Episodes")
    plt.legend()
    plt.show()





import cProfile
import pstats
if __name__ == '__main__':

    # 显示结果
    with open('result.txt', 'w') as f:
        p = pstats.Stats('output.prof', stream=f)
        p.sort_stats('cumtime').print_stats()

else:
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



    # 使用 cProfile 分析 my_function 的性能
    cProfile.run('run_tests()', 'output.prof')

