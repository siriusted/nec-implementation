from torch import nn
from torch.nn import functional as F

#TODO: confirm this is actually architecture used in DQN
DQN = lambda out_size: nn.Sequential(
        nn.Conv2d(4, 32, 8, stride=4, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.Flatten(),
        nn.Linear(3136, out_size),
        # nn.Linear(2592, 256),
        # nn.Linear(256, out_size)
)

# class DQN(nn.Module):
#     def __init__(self, out_size):
#         super().__init__()
#         self.conv1 = nn.Conv2d()
#         self.conv2 = nn.Conv2d()
#         self.fc1 = nn.Linear(2592, 256)
#         self.fc2 = nn.Linear(256, out_size)

#     def forward(self, x):
#         out = F.relu(self.conv1(x))
#         out = F.relu(self.conv2(out))
#         out = F.relu(self.fc1(out.flatten()))
#         out = self.fc2(out)

#         return out
