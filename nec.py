import torch
from torch import nn, optim
from dnd import DND

class NEC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_net = config['embedding_net']
        self.dnds = nn.ModuleList([DND() for _ in range(config['n_actions'])])

    def forward(self, state, action):
        """
        Forward pass through embedding CNN and DNDs

        The hope is that during optimization a single call to backward allows for
        backpropagation through the necessary dnd sub-modules

        TODO: check that backward in this manner doesn't affect parameters of other dnds not to be updated
        TODO: this should be able to work with a batch of actions so we can have less noisy
        gradient updates
        """
        h = self.embedding_net(state)
        dnd = self.dnds[action]
        q = dnd(h)

        return q

    def lookup(self, state):
        """
        To be used during environment interaction to get Q-values for a single state
        """
        with torch.no_grad():
            h = self.embedding_net(state)
            qs = [dnd.lookup(h) for dnd in self.dnds]

        return qs


if __name__ == "__main__":
    embedding_net = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 3))
    nec = NEC({"embedding_net": embedding_net, "n_actions": 2 })

    print(list(nec.parameters()))