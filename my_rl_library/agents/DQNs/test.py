import gymnasium as gym

from stable_baselines3 import DQN, DDPG, TD3, SAC, PPO
from stable_baselines3.common.monitor import Monitor
import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt


def train(env_name, Agent):
    # 创建 Gym 环境并添加 Monitor 以记录数据
    env = gym.make(env_name)

    env = Monitor(env, log_dir + "monitor.csv")

    # 初始化模型，启用 TensorBoard 日志记录
    model = Agent("MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log)

    # 开始训练
    model.learn(total_timesteps=1e5)
    model.save(log_dir + "agent_cartpole")
    env.close()

def test(env_name, Agent):
    # 加载模型
    env = gym.make(env_name)
    model = Agent.load("agent_cartpole")

    # 测试模型
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, done, _, __ = env.step(action)
        total_reward += reward
        env.render()

    print("Total Reward:", total_reward)
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
    discrete_envs_list = ["LunarLander-v2"]
    continuous_envs_list = ["Pendulum-v1"]
    continuous_agents_list = [DDPG,
                              TD3,
                              SAC,
                              PPO]

    for env_name in discrete_envs_list:
        Agent = DQN
        agent_name = str(Agent)
        train(env_name, Agent)
        # test(env_name, Agent)
        plot(env_name, agent_name)

    for env_name in continuous_envs_list:
        for Agent in continuous_agents_list:
            agent_name = str(Agent)
            train(env_name, Agent)
            # test(env_name, Agent)
            plot(env_name, agent_name)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Smoothed Training Rewards over Episodes")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    # 设置日志保存路径
    log_dir = "./logs/"
    tensorboard_log = "./agent_cartpole_tensorboard/"
    if os.path.exists(tensorboard_log):
        # 清空 log_dir 目录中的所有内容
        shutil.rmtree(tensorboard_log)
    # 创建空的 log_dir 目录
    os.makedirs(log_dir, exist_ok=True)

    run_tests()