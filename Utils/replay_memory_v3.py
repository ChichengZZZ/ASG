# import numpy as np
#
#
# class Replay_buffer():
#     '''
#     Code based on:
#     https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
#     Expects tuples of (state, next_state, action, reward, done)
#     '''
#     def __init__(self, max_size=50000):
#         self.storage = []
#         self.max_size = max_size
#         self.ptr = 0
#
#     def push(self, data):
#         if len(self.storage) == self.max_size:
#             self.storage[int(self.ptr)] = data
#             self.ptr = (self.ptr + 1) % self.max_size
#         else:
#             self.storage.append(data)
#
#     def sample(self, batch_size):
#         ind = np.random.randint(0, len(self.storage), size=batch_size)
#         x, y, u, r, d = [], [], [], [], []
#
#         for i in ind:
#             X, Y, U, R, D = self.storage[i]
#             x.append(np.array(X, copy=False))
#             y.append(np.array(Y, copy=False))
#             u.append(np.array(U, copy=False))
#             r.append(np.array(R, copy=False))
#             d.append(np.array(D, copy=False))
#
#         return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)



import random
from collections import deque, namedtuple
import numpy as np

# 定义 ReplayBuffer 类
# Experience = namedtuple('Experience',
#                         ['state', 'action', 'mask', 'next_state', 'reward', 'supervise_reward', 'supervise_state',
#                          'actions'])
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward', 'expert_reward', 'supervise_state', 'actions'))

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def push(self, *args):
        """Saves a transition."""
        self.buffer.append(Transition(*args))

    def __len__(self):
        return len(self.buffer)
