import queue
import heapq
import numpy as np
import torch

class SamplePool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.samples = None

    def put(self, sample, priority):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

class PriorityPool(SamplePool):
    def __init__(self, capacity):
        """
        初始化样本池，`capacity` 是池的最大容量
        优先级队列, 可以用于回溯更新
        """
        super().__init__(capacity)
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

class PrioritySamplePool(SamplePool):
    def __init__(self, capacity):
        """
        初始化样本池，`capacity` 是池的最大容量
        优先级采样.
        """
        super().__init__(capacity)
        self.samples = []  # 样本池中的样本列表
        self.counter = 0
    def put(self, sample, priority):
        """
        向样本池中添加一个新样本。最旧的样本将被挤出
        """
        if len(self.samples) < self.capacity:
            heapq.heappush(self.samples, (self.counter, priority, sample))
        else:
            # 样本池已满，替换掉最大优先级的样本
            heapq.heappushpop(self.samples, (self.counter, priority, sample))

        self.counter += 1
    def sample(self, batch_size):
        """
        根据优先级进行加权采样。
        """
        if not self.samples:
            return None

        # 获取所有样本的优先级，作为加权
        priorities = np.array([sample[1] for sample in self.samples])
        probabilities = priorities / priorities.sum()  # 转换为概率

        # 使用加权概率进行采样
        indices = torch.multinomial(torch.tensor(probabilities), batch_size, replacement=True)

        # 根据采样索引返回对应的样本
        sampled_samples = [self.samples[idx][2] for idx in indices]
        return torch.tensor(sampled_samples)
