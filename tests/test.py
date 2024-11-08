import heapq

class PriorityQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.heap = []  # 用于存储优先级队列的堆
        self.count = 0  # 用于生成插入顺序，打破优先级相同的元素的顺序

    def push(self, priority, item):
        """
        向优先级队列添加元素。如果队列已满，删除优先级最低的元素。
        """
        # 如果队列还没有满，直接插入
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (priority, self.count, item))
        else:
            # 如果队列已满，替换优先级最低的元素
            # 这里的 `heappushpop` 会插入新元素并弹出优先级最低的元素
            heapq.heappushpop(self.heap, (priority, self.count, item))

        self.count += 1

    def pop(self):
        """
        弹出并返回优先级最高的元素（优先级数值最小的元素）。
        """
        if self.heap:
            priority, count, item = heapq.heappop(self.heap)
            return item
        else:
            raise IndexError("pop from an empty priority queue")

    def peek(self):
        """
        返回优先级最高的元素（不删除）。
        """
        if self.heap:
            priority, count, item = self.heap[0]
            return item
        else:
            raise IndexError("peek from an empty priority queue")

    def size(self):
        """
        返回队列当前的大小。
        """
        return len(self.heap)

# 示例
pq = PriorityQueue(capacity=3)

# 插入元素
pq.push(2, 'item_2')
pq.push(1, 'item_1')
pq.push(3, 'item_3')

# 队列已满，插入新元素会挤出优先级最低的元素
pq.push(4, 'item_4')

# 弹出优先级最高的元素（优先级最小的元素）
print(pq.pop())  # 'item_1' (优先级 1)

# 队列大小
print(pq.size())  # 2

# 获取当前队列中的元素
while pq.size() > 0:
    print(pq.pop())
