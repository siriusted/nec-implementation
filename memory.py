import torch
import numpy as np

#TODO: Change to use a namedtuple
class ReplayBuffer:
    def __init__(self, observation_shape, capacity = 100000):
        self.capacity = capacity
        self.actions = np.empty((capacity, ), dtype = np.int64)
        self.observations = np.empty((capacity, ) + observation_shape, dtype = np.float32)
        self.returns = np.empty((capacity, ), dtype = np.float32)
        self.idx = 0
        self.effective_size = 0

    def append(self, observation, action, ret):
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.returns[self.idx] = ret
        self.idx = (self.idx + 1) % self.capacity
        self.effective_size = min(self.effective_size + 1, self.capacity)

    def append_batch(self, observations, actions, returns):
        batch_size = len(observations)
        idxs = np.arange(self.idx, self.idx + batch_size) % self.capacity
        self.observations[idxs] = observations
        self.actions[idxs] = actions
        self.returns[idxs] = returns
        self.idx = (self.idx + batch_size) % self.capacity
        self.effective_size = min(self.effective_size + batch_size, self.capacity)

    def sample(self, n = 1):
        idxs = np.random.randint(0, self.effective_size, size = n)
        return torch.from_numpy(self.observations[idxs]), self.actions[idxs], torch.from_numpy(self.returns[idxs])

    def size(self):
        """
        returns effective size of the replay buffer
        """
        return self.effective_size
