import torch
from torch import nn, optim
from dnd import DND

class NEC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_net = config['embedding_net']
        self.dnds = nn.ModuleList([DND() for _ in range(config['n_actions'])])

    def forward(self, states, actions):
        """
        Forward pass through embedding CNN and DNDs

        During optimization a single call to backward allows for
        backpropagation through the necessary dnd sub-modules

        TODO: check that backward in this manner doesn't affect parameters of other dnds not to be updated
        """
        keys = self.embedding_net(states)
        q = torch.tensor([self.dnds[action.item()](key) for key, action in zip(keys, action)], requires_grad=True) # not taking advantage of batch forward capability of torch
        # instead group keys by dnd
        # call forward on each dnd with group of keys

        return q

    def lookup(self, obs):
        """
        To be used during environment interaction to get Q-values for a single state
        """
        with torch.no_grad():
            key = self.embedding_net(obs)
            qs = [dnd.lookup(key) for dnd in self.dnds]

            return qs, key


if __name__ == "__main__":
    embedding_net = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 3))
    nec = NEC({"embedding_net": embedding_net, "n_actions": 2 })

    print(list(nec.parameters()))