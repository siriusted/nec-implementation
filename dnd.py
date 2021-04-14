import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils # allows utilizing torch tensors as input to faiss indexes and functions
from torch import nn
from torch.nn import functional as F, Parameter

def torch_replace_knn(xq, xb, k, metric=faiss.METRIC_L2):
    if type(xb) is np.ndarray:
        return faiss.knn_numpy(xq, xb, k, metric)

    #TO BE COMPLETED later for fun
    # for now just call .numpy() on torch tensors

def _inverse_distance_kernel(sq_distances, delta = 1e-3):
    """
    Kernel used in Pritzel et. al, 2017
    """
    return 1 / (sq_distances + delta)

def _knn_search(queries, data, k):
    """
    Perform exact knn search (should be replaced with approximate and extended to utilize GPU)

    Return the k nearest keys
    """
    if type(queries) is torch.Tensor:
        queries, data = queries.detach().numpy(), data.detach().numpy()

    return faiss.knn(queries, data, k) # (distances, indexes)

def _combine_by_key(keys, values, op):
    """
    Combines duplicate keys' values using the operator op (max or mean)
    """
    keys = [tuple(key.detach().numpy()) for key in keys]
    ks, vs = [], []
    key_map = {}

    if op == 'max':
        for i, key in enumerate(keys):
            if key in key_map:
                idx = key_map[key]
                old_val = vs[idx]
                vs[idx] = values[i] if old_val < values[i] else old_val
            else:
                key_map[key] = len(ks)
                ks.append(key)
                vs.append(values[i])
    elif op == 'mean':
        for i, key in enumerate(keys):
            if key in key_map:
                # update average using stored average, running count, and new value
                idx, n = key_map[key]
                vs[idx] = (vs[idx] * n + values[i]) / (n + 1)
                key_map[key][1] += 1
            else:
                key_map[key] = [len(ks), 1] # store idx in new arrays and running count
                ks.append(key)
                vs.append(values[i])

    return np.array(ks), np.array(vs)

class DND(nn.Module):
    """
    Implementation of a differentiable neural dictionary as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)
    """

    def __init__(self, config):
        super().__init__()

        self.capacity = config['capacity']
        self.num_neighbours = config['neighbours']
        self.key_size = config['key_size']
        self.alpha = config['alpha']

        self.keys = Parameter(torch.ones(self.capacity, self.key_size) * 1e8) # use very large values to allow for low similarity with keys while warming up
        self.values = Parameter(torch.zeros(self.capacity))

        self.keys_hash = { tuple(self.keys[0].detach().numpy()): 0 } # one idea is to use actual index as value in this hash
        self.lru_list = np.linspace(1, self.capacity, self.capacity)

    def lookup(self, key):
        """
        To be used wkeyen going through DND without plans to keep gradients

        Params:
            - key: state embedding to be looked up

        Steps:
        1. find k nearest neighbours of key
        2. compute Q with k nns
        3. maintain LRU neighbours idxs
        4. return desired Q
        """
        with torch.no_grad():
            sq_distances, neighbour_idxs = _knn_search(key, self.keys, self.num_neighbours)

            # maintain lru here
            # all neighbour_idxs should be set to recently used than all others

            # compute the actual Q
            w_i = _inverse_distance_kernel(torch.tensor(sq_distances))
            w_i /= torch.sum(w_i)
            v_i = self.values[neighbour_idxs]

            return torch.sum(w_i * v_i).item()

    def forward(self, keys):
        """
        Here we're along the gradient pathway

        during backward from an outer module,
        self.keys and self.values involved in computing Q are updated

        1. compute Q with knns
        2. return Q
        """

        sq_distances, neighbour_idxs = _knn_search(keys, self.keys, self.num_neighbours)

        # maintain lru_list here: flatten then pick unique neighbour indices

        # re-compute distances for backprop
        neighbours = self.keys[neighbour_idxs.reshape(-1)].view(-1, self.num_neighbours, self.key_size)
        sq_distances = ((keys.unsqueeze(dim = 1) - neighbours) ** 2).sum(dim = 2)
        weights = _inverse_distance_kernel(sq_distances)
        weights /= weights.sum(dim = 1, keepdim = True)

        values = self.values[neighbour_idxs.reshape(-1)].view(-1, self.num_neighbours, 1)

        return torch.sum(weights.unsqueeze(dim = 2) * values, dim = 1)


    def update_batch(self, keys, values):
        """
        Update the DND with keys and values experienced from an episode
        """
        # first handle duplicates inside the batch of data by either taking the max or averaging
        keys, values = _combine_by_key(keys, values, 'max') # returns keys as a tuple that can be used in keys_hash
        match_idxs, match_dnd_idxs, new_idxs = [], [], []

        # then group indices of exact matches and new keys
        for i, key in enumerate(keys):
            if key in self.keys_hash:
                match_dnd_idxs.append(self.keys_hash[key])
                match_idxs.append(i)
            else:
                new_idxs.append(i)

        num_matches, num_new = len(match_idxs), len(new_idxs)
        with torch.no_grad():
            # update exact matches using dnd learning rate
            if num_matches:
                self.values[match_dnd_idxs] += self.alpha * (values[match_idxs] - self.values[match_dnd_idxs])
                # make the associated dnd idxs MRU

            if num_new:
                lru_idxs = self.lru_list[: ]
                self.keys[lru_idxs] = keys[new_idxs]
                self.values[lru_idxs] = values[new_idxs]
                # move to MRU position just before matches

                #update self.keys_hash


    def update(self):
        """
        This is a function to be called to update the index used for approximate search
        """

if __name__ == "__main__":
    #TODO: write unit tests
    dnd = DND({ "capacity": 5, "neighbours": 5, "key_size": 3, "alpha": 0.5 })
    print(list(dnd.parameters()))
    print("Testing with torch tensors")
    data = torch.tensor([[1, 1], [2, 2], [8, 8], [9, 9]], dtype=torch.float32)
    queries = torch.tensor([[3, 3], [5, 5], [7, 7]], dtype=torch.float32)

    dists, idxs = _knn_search(queries, data, 2)
    print(dists)
    print(idxs)
    print(type(dists), type(idxs))

    print("Testing with numpy arrays")
    dists, idxs = _knn_search(queries.numpy(), data.numpy(), 2)
    print(dists)
    print(idxs)
    print(type(dists), type(idxs))
