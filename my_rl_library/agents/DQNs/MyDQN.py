import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

from my_rl_library.agents.Agent import Agent
from my_rl_library.utils import *


class DQNReplayBuffer(PrioritySamplePool):
    def sample(self, batch_size):
        samples = super().sample(batch_size)

        states, actions, rewards, next_states, dones = zip(*samples)

        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.int64)

        important_weights = np.ones_like(dones, dtype=np.int64)

        return states, actions, rewards, next_states, dones, important_weights


class MyDQN(Agent):
    def __init__(self, name, env, hidden_dims=None):
        super().__init__("MyDQN", env)

        if hidden_dims is None:
            hidden_dims = [64, 64]
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        self.tar_model = MyFCNN(state_dim, action_dim, hidden_dims)
        self.policy_model = MyFCNN(state_dim, action_dim, hidden_dims)
        self.tar_model.load_state_dict(self.policy_model.state_dict())
        self.tar_model.eval()

    def predict(self, state, deterministic=True):
        super().predict(state, deterministic)
        action = self.tar_model(torch.tensor(state, dtype=torch.float)).argmax().item()
        return action

    def learn(self, total_timesteps=1e5):
        self.train(total_timesteps=1e5)

    def train(self, total_timesteps=1e5, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                  buffer_capacity=10000, batch_size=64, learning_rate=0.001, replay_period=128):
        policy_net = self.policy_model
        target_net = self.tar_model
        env = self.env

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        replay_buffer = DQNReplayBuffer(buffer_capacity)
        epsilon = epsilon_start

        for step_i in range(int(total_timesteps)):
            # 使用 epsilon-greedy 策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(state, dtype=torch.float)).argmax().item()

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.put((state, action, reward, next_state, done), 1)

            state = next_state
            total_reward += reward

            if done:
                state, _ = env.reset()
                total_reward = 0

                # 更新 epsilon
                epsilon = max(epsilon_end, epsilon_decay * epsilon)

                print(f"Episode {step_i}, Total Reward: {total_reward}")

            # 训练 Q 网络
            if len(replay_buffer) >= batch_size and step_i % replay_period == 0:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = tuple(map(torch.tensor, (states, actions, rewards, next_states, dones)))

                # 计算目标 Q 值
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 每隔一定步数更新目标网络
                if step_i % (replay_period * 10) == 0:
                    target_net.load_state_dict(policy_net.state_dict())


        return policy_net


