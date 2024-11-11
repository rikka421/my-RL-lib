import torch
import torch.optim as optim
import random

from my_rl_library.agents.Agent import Agent
from my_rl_library.utils import *


class DQNReplayBuffer(PrioritySamplePool):
    def sample(self, batch_size, device):
        with torch.no_grad():
            # 采样得到的是一个[(s, a, r, s, d), (s, a, r, s, d), ...]
            samples, probabilities = super().sample(batch_size, device)

            states, actions, rewards, next_states, dones = zip(*samples)

            states = torch.tensor(np.array(states), dtype=torch.float32).to(device=device)
            actions = torch.tensor(np.array(actions), dtype=torch.int64).view(-1, 1).to(device=device)
            next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device=device)
            rewards = torch.tensor(np.array(rewards), dtype=torch.float32).view(-1, 1).to(device=device)
            dones = torch.tensor(np.array(dones), dtype=torch.int64).view(-1, 1).to(device=device)
            probabilities = torch.tensor(np.array(probabilities), dtype=torch.float32).view(-1, 1).to(device=device)
            return states, actions, rewards, next_states, dones, probabilities

    def update_samples(self, priorities):
        with torch.no_grad():
            for i in range(len(priorities)):
                index = self.indices[i]
                self.samples[index][1] = priorities[i].item()
                self.max_priority = max(self.max_priority, self.samples[index][1])


class MyDQN(Agent):
    def __init__(self, name, env, hidden_dims=None, double_q=False, dueling_q=False, priority_pool=True, device="cpu"):
        super().__init__("MyDQN", env)

        self.priority_pool = priority_pool
        self.double_q = double_q
        self.dueling_q = dueling_q
        self.device = device

        if hidden_dims is None:
            hidden_dims = [64, 64]
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.env = env
        if dueling_q:
            pass
        else:
            self.target_network = MyFCNN(state_dim, action_dim, hidden_dims).to(device)
            self.action_network = MyFCNN(state_dim, action_dim, hidden_dims).to(device)
        self.target_network.load_state_dict(self.action_network.state_dict())
        self.target_network.eval()

    def predict(self, state, deterministic=True):
        super().predict(state, deterministic)
        action = self.target_network(torch.tensor(state, dtype=torch.float)).argmax().item()
        return action

    def learn(self, total_timesteps=1e5):
        self.train(total_timesteps=1e5)

    def step(self, buffer, optimizer, batch_size, gamma, beta):
        # 采样得到一个tensor. probabilities 是一个tensor
        states, actions, rewards, next_states, dones, probabilities = buffer.sample(batch_size, self.device)
        q_values = self.action_network(states).gather(1, actions)
        q_values_next = self.target_network(next_states)
        if not self.double_q:
            target_q = rewards + gamma * torch.max(q_values_next, dim=1).values.view(-1, 1) * (1 - dones)
        else:
            # 如果是Double DQN
            best_actions = torch.argmax(self.action_network(next_states), dim=1).view(-1, 1) # 选择动作, 通过action
            target_q = rewards + gamma * q_values_next.gather(1, best_actions).view(-1, 1) * (1 - dones)  # 估值动作, 通过target

        errors = q_values - target_q
        # print(torch.sum(self.action_network(states)), torch.sum(self.action_network(states) - self.target_network(states)) )
        if not self.priority_pool:
            loss = torch.sum((errors * errors))
            # print(errors, loss)
        else:
            # 若引入优先级采样和重要性采样
            weights = torch.pow(len(buffer) * probabilities, -beta)
            loss = torch.sum(weights * (errors * errors))
            new_priorities = torch.abs(errors)
            buffer.update_samples(new_priorities) # 将新计算出的priorities更新到样本池中

        # step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, total_timesteps=1e5, gamma=0.99, epsilon_start=0.5, epsilon_end=0.05, epsilon_decay=0.995,
                  buffer_capacity=10000, batch_size=64, learning_rate=0.0001, replay_period=1, alpha=0.9, beta=0.9):
        policy_net = self.action_network
        target_net = self.target_network
        env = self.env

        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        replay_buffer = DQNReplayBuffer(buffer_capacity, alpha)
        epsilon = epsilon_start

        state, _ = env.reset()
        total_reward = 0
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
            replay_buffer.put([state, action, reward, next_state, done])

            state = next_state
            total_reward += reward

            if done:
                # 更新 epsilon
                epsilon = max(epsilon_end, epsilon_decay * epsilon)
                print(f"Step {step_i}, Total Reward: {total_reward}, epsilon: {epsilon :.2f}")

                state, _ = env.reset()
                total_reward = 0

            # 训练 Q 网络
            if len(replay_buffer) >= batch_size and step_i % replay_period == 0:
                # 每隔一定步数采样进行训练 (步数量级为1)
                self.step(replay_buffer, optimizer, batch_size, gamma, beta)
            if step_i % 1000 == 0:
                # 每隔一定步数更新目标网络 (步数量级为1000)
                target_net.load_state_dict(policy_net.state_dict())


        return policy_net


