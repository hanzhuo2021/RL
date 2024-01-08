import random
from collections import deque

import numpy as np


# 重播缓冲区:这是智能体以前的经验， 为了使算法具有稳定的行为，重播缓冲区应该足够大以包含广泛的体验。
# 如果仅使用最新数据，则可能会过分拟合，如果使用过多的经验，则可能会减慢模型的学习速度。 这可能需要一些调整才能正确。
class Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()
