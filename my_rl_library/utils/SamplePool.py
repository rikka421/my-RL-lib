import queue
import heapq
import numpy as np
import torch


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
    def __init__(self, capacity, alpha, epsilon=1e-4):
        """
        初始化样本池，`capacity` 是池的最大容量
        优先级采样. 存储通过列表和数组, 添加通过heapq, 采样通过torch.tensor
        """
        self.capacity = capacity
        self.samples = []  # 样本池中的样本列表
        self.counter = 0
        self.alpha = alpha

        self.max_priority = epsilon
    def put(self, sample):
        """
        向样本池中添加一个新样本。最旧的样本将被挤出
        实际上通过优先级实现了先入先出队列
        """
        if len(self.samples) < self.capacity:
            heapq.heappush(self.samples, [self.counter, self.max_priority, sample])
        else:
            # 样本池已满，替换掉最大优先级的样本
            heapq.heappushpop(self.samples, [self.counter, self.max_priority, sample])

        self.counter += 1
    def sample(self, batch_size, device):
        """
        根据优先级进行加权采样。
        """
        if not self.samples:
            return None

        # 获取所有样本的优先级，作为加权
        priorities = torch.tensor([sample[1] ** self.alpha for sample in self.samples], dtype=torch.float32)
        probabilities = torch.nn.functional.normalize(priorities, p=1, dim=0) # 转换为概率

        # 使用加权概率进行采样
        # indices = np.random.choice(range(len(self.samples)), size=batch_size, p=probabilities, replace=True)
        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        self.indices = indices

        # 根据采样索引返回对应的样本
        sampled_samples = [self.samples[idx][2] for idx in indices]
        return sampled_samples, probabilities[indices]

    def __len__(self):
        return len(self.samples)

