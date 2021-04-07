import torch
import numpy as np

#TODO: Change to use a namedtuple
class ReplayBuffer:
    def __init__(self, capacity = 100000):
        self.capacity = capacity
        self.actions = np.empty((capacity, ), dtype = np.uint8) # assuming not more than 255 actions
        self.observations = np.empty((capacity, ), dtype = np.float32)
        self.returns = np.empty((capacity, ), dtype = np.float32)
        self.idx = 0
        self.count = 0

    def append(self, observation, action, ret):
        self.observations[self.idx] = observation
        self.actions[self.idx] = action
        self.returns[self.idx] = ret
        self.idx = (self.idx + 1) % self.capacity
        self.count += 1 # Think of a different way to do this, it will overflow at some point in a long enough training period

    def append_batch(self, observations, actions, returns):
        batch_size = len(observations)
        idxs = np.arange(self.idx, self.idx + batch_size) % self.capacity
        self.observations[idxs] = observations
        self.actions[idxs] = actions
        self.returns[idxs] = returns
        self.idx = (self.idx + batch_size) % self.capacity
        self.count += batch_size

    def sample(self, n = 1):
        idxs = np.random.randint(0, self.capacity if self.count >= self.capacity else self.idx, size = n)
        return torch.from_numpy(self.observations[idxs]), torch.from_numpy(self.actions[idxs]), torch.from_numpy(self.returns[idxs])

if __name__ == "__main__":
    #TODO: write unit tests
    pass
