import queue
import numpy as np
import torch
import heapq

class CircularBuffer:
    def __init__(self, size, dtype=np.float32):
        self.size = size
        self.buffer = np.zeros(size, dtype=dtype)  # 初始化为0，可以指定其他类型
        self.index = 0
        self.length = 0
        self.capacity = self.size[0]

    def append(self, value):
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.capacity
        if self.length < self.capacity:
            self.length += 1

    def get(self):
        # 返回按顺序的列表视图
        # 在这里并不关心实际的先后顺序
        if self.length == self.capacity:
            return self.buffer
        else:
            return self.buffer[:self.index]

    def __getitem__(self, idx):
        # 支持按顺序访问
        if idx >= self.length:
            raise IndexError("index out of range")
        return self.buffer[idx]

    def __setitem__(self, idx, value):
        if idx >= self.length:
            raise IndexError("index out of range")
        self.buffer[idx] = value

class PriorityPool():
    def __init__(self, capacity):
        """
        初始化样本池，`capacity` 是池的最大容量
        优先级队列, 可以用于回溯更新
        """
        self.capacity = capacity
        if capacity != -1:
            self.samples = queue.PriorityQueue(maxsize=capacity)
        else:
            self.samples = queue.PriorityQueue()

    def put(self, sample, priority):
        """
        向样本池中添加一个新样本。高优先级的样本被挤出
        """
        if self.samples.full():
            self.samples.get()
        self.samples.put((priority, sample))

    def get(self):
        """
        获取优先级最高的样本
        """
        if self.samples.empty():
            return None
        return self.samples.get()


class PrioritySamplePool():
    def __init__(self, capacity, alpha, states_dim, actions_dim):
        """
        初始化样本池，`capacity` 是池的最大容量
        优先级采样. 存储通过列表和数组, 添加通过heapq, 采样通过torch.tensor
        """
        self.capacity = capacity
        self.priorities = CircularBuffer((capacity, ), dtype=np.float32)
        self.states = CircularBuffer((capacity, states_dim), dtype=np.float32)
        self.actions = CircularBuffer((capacity, actions_dim), dtype=np.int64)
        self.rewards = CircularBuffer((capacity, ), dtype=np.float32)
        self.next_states = CircularBuffer((capacity, states_dim), dtype=np.float32)
        self.dones = CircularBuffer((capacity, ), dtype=np.int64)

        self.alpha = alpha
        self.max_priority = 1.0
    def put(self, state, action, reward, next_state, done):
        """
        向样本池中添加一个新样本。最旧的样本将被挤出
        实际上通过优先级实现了先入先出队列
        """
        self.priorities.append(self.max_priority)
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)

    def sample(self, batch_size, device):
        """
        根据优先级进行加权采样。
        """
        if self.priorities.length == 0:
            return None

        with torch.no_grad():
            # 获取所有样本的优先级，作为加权
            priorities = self.priorities.get()
            priorities = np.pow(priorities, self.alpha)
            probabilities = priorities / np.sum(priorities) # 转换为概率
            # priorities = torch.tensor([sample[1] ** self.alpha for sample in self.samples], dtype=torch.float32)
            # probabilities = torch.nn.functional.normalize(priorities, p=1, dim=0) # 转换为概率

            # 使用加权概率进行采样
            indices = np.random.choice(range(self.priorities.length), size=batch_size, p=probabilities, replace=True)
            # indices = torch.multinomial(probabilities, batch_size, replacement=True)
            self.indices = indices

            # 根据采样索引返回对应的样本
            states = self.states.get()[indices]
            actions = self.actions.get()[indices]
            rewards = self.rewards.get()[indices]
            next_states = self.next_states.get()[indices]
            dones = self.dones.get()[indices]

            return states, actions, rewards, next_states, dones, probabilities[indices]

    def __len__(self):
        return self.priorities.length

