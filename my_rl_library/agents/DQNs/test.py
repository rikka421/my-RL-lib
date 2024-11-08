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


def train(env_name, Agent):
    # 创建 Gym 环境并添加 Monitor 以记录数据
    env = gym.make(env_name)

    env = Monitor(env, log_dir + "monitor.csv")

    # 初始化模型，启用 TensorBoard 日志记录
    model = Agent("MlpPolicy", env)# , verbose=1, tensorboard_log=tensorboard_log)

    # 开始训练
    model.learn(total_timesteps=1e5)
    model.save(log_dir + "model_weights")
    env.close()

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
        train(env_name, MyDQN)
        plot(env_name, "MyDQN")
        # train(env_name, DQN)
        # plot(env_name, "DQN")

    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Smoothed Training Rewards over Episodes")
    plt.legend()
    plt.show()



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