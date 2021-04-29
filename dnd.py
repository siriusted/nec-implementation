import numpy as np
import torch
import faiss
import faiss.contrib.torch_utils # allows utilizing torch tensors as input to faiss indexes and functions
from torch import nn
from torch.nn import functional as F, Parameter

def _inverse_distance_kernel(sq_distances, delta = 1e-3):
    """
    Kernel used in Pritzel et. al, 2017
    """
    # https://discuss.pytorch.org/t/runtimeerror-function-sqrtbackward-returned-nan-values-in-its-0th-output/48702
    return 1 / (torch.sqrt(sq_distances + 1e-8) + delta)

def _knn_search(queries, data, k):
    """
    Perform exact knn search (should be replaced with approximate)

    Return the k nearest keys
    """
    if torch.cuda.is_available(): # not the best way but should let me know that gpu is being used
        res = faiss.StandardGpuResources()
        D, I = faiss.knn_gpu(res, queries, data, k)
        return D.detach().cpu().numpy(), I.detach().cpu().numpy()

    queries, data = queries.detach().numpy(), data.detach().numpy()
    return faiss.knn(queries, data, k) #(distances, indices)

def _combine_by_key(keys, values, op):
    """
    Combines duplicate keys' values using the operator op (max or mean)
    """
    keys = [tuple(key) for key in keys.detach().cpu().numpy()]
    ks, vs = [], []
    key_map = {}

    #TODO: to keep order of recency maybe cycle positions when duplicates
    # are encountered so the key remains the latest in the array order\
    # but does it matter !!???
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

    return ks, vs

class DND(nn.Module):
    """
    Implementation of a differentiable neural dictionary as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)
    """

    def __init__(self, config):
        super().__init__()

        self.capacity = config['dnd_capacity']
        self.num_neighbours = config['num_neighbours']
        self.key_size = config['key_size']
        self.alpha = config['alpha']
        self.device = config['device']

        # opposed to paper description, this list is not growing but pre-initialised and gradually replaced
        self.keys = Parameter(torch.ones(self.capacity, self.key_size, device=config['device']) * 1e6) # use large values to allow for low similarity with keys while warming up
        self.values = Parameter(torch.zeros(self.capacity, device=config['device']))

        self.keys_hash = {} # one idea is to use actual index as value in this hash
        self.last_used = np.linspace(self.capacity, 1, self.capacity, dtype=np.uint32) # used to manage lru replacement
        # initialised in descending order to foster the replacement of earlier indexes before later ones

    def lookup(self, key) -> float:
        """
        To be used when going through DND without plans to keep gradients

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
            self.last_used += 1 # increment time last used for all keys
            self.last_used[neighbour_idxs.reshape(-1)] = 0 # reset time last used for neighbouring keys

            # compute the actual Q
            w_i = _inverse_distance_kernel(torch.tensor(sq_distances, device=self.device))
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

        neighbour_idxs = neighbour_idxs.reshape(-1) # flattened list view useful below

        # maintain lru here
        self.last_used += 1
        self.last_used[neighbour_idxs] = 0

        # re-compute distances for backprop
        neighbours = self.keys[neighbour_idxs].view(-1, self.num_neighbours, self.key_size)
        sq_distances = ((keys.unsqueeze(dim = 1) - neighbours) ** 2).sum(dim = 2)
        weights = _inverse_distance_kernel(sq_distances)
        weights /= weights.sum(dim = 1, keepdim = True)

        values = self.values[neighbour_idxs].view(-1, self.num_neighbours)

        return torch.sum(weights * values, dim = 1)


    def update_batch(self, keys, values):
        """
        Update the DND with keys and values experienced from an episode
        """
        # first handle duplicates inside the batch of data by either taking the max or averaging
        keys, values = _combine_by_key(keys, values, 'max') # returns keys as a list of tuples that can be used to index self.keys_hash
        match_idxs, match_dnd_idxs, new_idxs = [], [], []

        # probably limit to make sure keys and values are not larger than capacity

        # then group indices of exact matches and new keys
        for i, key in enumerate(keys):
            if key in self.keys_hash:
                match_dnd_idxs.append(self.keys_hash[key])
                match_idxs.append(i)
            else:
                new_idxs.append(i)

        num_matches, num_new = len(match_idxs), len(new_idxs)

        self.last_used += 1 # maintain time since keys used

        with torch.no_grad():
            # make tensors for fancy indexing and easy interoperability with self.keys and self.values
            keys, values = torch.tensor(keys, device=self.device), torch.tensor(values, device=self.device)

            # update exact matches using dnd learning rate
            if num_matches:
                self.values[match_dnd_idxs] += self.alpha * (values[match_idxs] - self.values[match_dnd_idxs])
                self.last_used[match_dnd_idxs] = 0

            # replace least recently used keys with new keys
            if num_new:
                lru_idxs = np.argsort(self.last_used)[-num_new:] # get lru indices using the self.last_used
                self.keys[lru_idxs] = keys[new_idxs]
                self.values[lru_idxs] = values[new_idxs]
                self.last_used[lru_idxs] = 0

                # update self.keys_hash
                inv_hash = {v: k for k, v in self.keys_hash.items()}

                for idx in lru_idxs:
                    if idx in inv_hash:
                        del self.keys_hash[inv_hash[idx]]
                    self.keys_hash[tuple(self.keys[idx].detach().cpu().numpy())] = idx


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
