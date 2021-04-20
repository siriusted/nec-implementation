import torch
from torch import nn, optim
from dnd import DND

class NEC(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embedding_net = config['embedding_net']
        self.dnds = nn.ModuleList([DND(config) for _ in range(config.num_actions)])

    def forward(self, observations):
        """
        Forward pass through embedding CNN and DNDs

        During optimization a single call to backward allows for
        backpropagation through the necessary dnd sub-modules

        TODO: check that backward in this manner doesn't affect parameters of other dnds not to be updated
        """
        keys = self.embedding_net(observations)
        qs = torch.stack([dnd(keys) for dnd in self.dnds]).T # to get q_values in shape [batch_size x actions]

        return qs

    def lookup(self, obs):
        """
        To be used during environment interaction to get Q-values for a single state
        """
        with torch.no_grad():
            key = self.embedding_net(obs)
            qs = [dnd.lookup(key) for dnd in self.dnds]

            return qs, key

    def update_memory(self, action, keys, values):
        """
        Used to batch update an action's DND at the end of an episode
        """
        self.dnds[action].update_batch(keys, values)


if __name__ == "__main__":
    embedding_net = nn.Sequential(nn.Linear(5, 5), nn.Linear(5, 3))
    nec = NEC({"embedding_net": embedding_net, "n_actions": 2 })

    print(list(nec.parameters()))