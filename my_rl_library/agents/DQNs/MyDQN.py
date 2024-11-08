import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

env = gym.make("CartPole-v1")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*sample)
        return (
            torch.tensor(states, dtype=torch.float),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float),
            torch.tensor(next_states, dtype=torch.float),
            torch.tensor(dones, dtype=torch.float),
        )

    def __len__(self):
        return len(self.buffer)

def train_dqn(env, num_episodes=500, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_capacity=10000, batch_size=64, learning_rate=0.001):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for t in range(1000):  # 每个 episode 的最大步数
            # 使用 epsilon-greedy 策略选择动作
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = policy_net(torch.tensor(state, dtype=torch.float)).argmax().item()

            # 执行动作
            next_state, reward, done, _, __ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

            # 训练 Q 网络
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # 计算目标 Q 值
                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                next_q_values = target_net(next_states).max(1)[0]
                target_q_values = rewards + gamma * next_q_values * (1 - dones)

                loss = nn.functional.mse_loss(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # 每隔一定步数更新目标网络
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 更新 epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        print(f"Episode {episode}, Total Reward: {total_reward}")

    return policy_net


trained_policy = train_dqn(env)

state = env.reset()
done = False
total_reward = 0

while not done:
    action = trained_policy(torch.tensor(state, dtype=torch.float)).argmax().item()
    state, reward, done, _ = env.step(action)
    total_reward += reward
    env.render()

print("Total Reward:", total_reward)
env.close()
