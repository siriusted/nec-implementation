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

def _inverse_distance_kernel(h1, h2, delta = 1e-3):
    """
    Kernel used in Pritzel et. al, 2017
    """
    return 1 / (torch.dist(h1, h2) + delta)

def _knn_search(queries, data, k):
    """
    Perform exact knn search (should be replaced with approximate and extended to utilize GPU)

    Return the k nearest keys
    """
    if type(queries) is torch.Tensor:
        queries, data = queries.numpy(), data.numpy()

    return faiss.knn(queries, data, k) # (distances, indexes)

class DND(nn.Module):
    """
    Implementation of a differentiable neural dictionary as introduced in
    Neural Episodic Control (Pritzel et. al, 2017)
    """

    def __init__(self, config):
        super(DND, self).__init__()

        self.capacity = config['capacity']
        self.num_neighbours = config['neighbours']
        self.key_size = config['key_size']
        self.alpha = config['alpha']

        self.keys = Parameter(torch.ones(self.capacity, self.key_size) * 1e8) # use very large values to allow for low similarity with keys while warming up
        self.values = Parameter(torch.zeros(self.capacity))

        self.keys_hash = { tuple(self.keys[0].detach().numpy()): 0 } # one idea is to use actual index as value in this hash


    def lookup(self, h):
        """
        To be used when going through DND without plans to keep gradients

        Params:
            - h: state embedding to be looked up

        Steps:
        1. find k nearest neighbours of h (k := min(num_neighbours, len(keys/values)))
        2. compute Q with k nns
        3. maintain LRU neighbours idxs
        4. call write with h and q
        5. return desired Q
        """
        with torch.no_grad():
            distances, neighbour_idxs = _knn_search(h, self.keys, self.num_neighbours)

            # maintain lru here
            # all neighbour_idxs should be set to recently used than all others

            # compute the actual Q
            w_i = torch.cat([_inverse_distance_kernel(h, h_i) for h_i in self.keys[neighbour_idxs]])
            w_i /= torch.sum(w_i)
            v_i = self.values[neighbour_idxs]

            return torch.sum(w_i * v_i)

    def write(self, h, q):
        """
        To be called during lookup without gradients

        Params:
            - h: embedding to be written
            - q: value to be written

        and probably also during training (naa, I think during training we should not do this,
        rather we should find a way to allow the contents in memory to naturally change with backprop)

        Steps:
        1. is it existing?
        2. if yes, update with q-learning update with high alpha
        3. if no, insert q, paying attention to possible eviction based on LRU neighbours

        This logic will not exist alone but in batch
        """
        pass

    def update_batch(self, keys, values):
        """
        So how do we do this?

        What information is needed?

        Keys looked up during the episode
        Values/returns computed for these keys

        least recently used neighbours so we can remove them
        """

        # maybe its better to build these two in one for loop, test both for speed later on
        # this is wrong because we need two indexes
        # 1. for the index in the arguments passed in
        # 2. for the index in the key hash
        existing_idxs = np.asarray([ self.keys_hash[key] for key in keys if key in self.keys_hash ])
        non_existing_idxs = np.asarray(
            list(
                set(np.arange(self.capacity)) - set(existing_idxs)
            )
        )


        if len(existing_idxs):
            self.values[existing_idxs] += self.alpha * (values[some_other_idx] - self.values[existing_idxs])

        if len(non_existing_idxs):
            lru_idxs = np.zeros((20, ))
            self.keys[lru_idxs] = keys[non_existing_idxs]
            self.values[lru_idxs] = values[non_existing_idxs]

            #update keys of lru idxs in keys_hash
            #maybe update lru

    def forward(self, h):
        """
        Here we're along the gradient pathway

        The hope is that during backward from an outer module,
        self.keys and self.values involved in computing Q are updated

        1. compute Q with knns
        2. return Q
        """

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
