import torch
from torch import nn
from torch.nn import functional as F

#DQN from https://github.com/Kaixhin/EC
DQN_EC = lambda out_size: nn.Sequential(
        nn.Conv2d(4, 32, kernel_size = 8, stride = 4, padding = 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size = 4, stride = 2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size = 3),
        nn.Flatten(),
        nn.Linear(3136, out_size),
)

# Original DQN architecture
class DQN(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 4, stride = 2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, out_size)

    def forward(self, x):
        hidden = F.relu(self.conv1(x))
        hidden = F.relu(self.conv2(hidden))
        hidden = F.relu(self.fc1(hidden.flatten(start_dim = 1)))
        output = self.fc2(hidden)

        return output

# simple MLP for Cartpole
MLP = lambda out_size: nn.Sequential(
    nn.Linear(4, 24),
    nn.Linear(24, out_size),
)

if __name__ == "__main__":
    dqn = DQN_EC(4)
    inp = torch.rand(2, 4, 84, 84)

    out1 = dqn(inp)

    assert out1.shape == torch.Size([2, 4])

    dqn_orig = DQN(4)
    out2 = dqn_orig(inp)

    assert out2.shape == torch.Size([2, 4])