from torch import nn
from torch.nn import functional as F

DQN = lambda out_size: nn.Sequential(
        nn.Conv2d(),
        nn.ReLU(),
        nn.Conv2d(),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(2592 , 256),
        nn.Linear(256, out_size)
)

class DQN(nn.Module):
    def __init__(self, out_size):
        super().__init__()
        self.conv1 = nn.Conv2d()
        self.conv2 = nn.Conv2d()
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, out_size)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.fc1(out.flatten()))
        out = self.fc2(out)

        return out
