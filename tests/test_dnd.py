import torch
import numpy as np
from dnd import _combine_by_key

def test_combine_by_key():
    # no duplicates + max
    keys = torch.stack([
        torch.tensor([1., 1.]),
        torch.tensor([2., 2.]),
        torch.tensor([3., 3.])
    ])

    values = np.array([1., 2., 3.], dtype = np.float32)

    ks, vs = _combine_by_key(keys, values, 'max')

    assert ks == [(1., 1.), (2., 2.), (3., 3.)]
    assert vs == [1., 2., 3.]

    #  no duplicates + mean
    ks, vs = _combine_by_key(keys, values, 'mean')

    assert ks == [(1., 1.), (2., 2.), (3., 3.)]
    assert vs == [1., 2., 3.]

    # duplicates + max
    keys_with_duplicates = torch.stack([
        torch.tensor([1., 1.]),
        torch.tensor([2., 2.]),
        torch.tensor([1., 1.]),
        torch.tensor([3., 3.]),
        torch.tensor([3., 3.]),
        torch.tensor([5., 5.]),
        torch.tensor([2., 2.]),
        torch.tensor([1., 1.])
    ])

    values_for_duplicates = np.array([10., 5., 20., 9., 3., 7., 15., 18.], dtype = np.float32)

    ks, vs = _combine_by_key(keys_with_duplicates, values_for_duplicates, 'max')

    assert ks == [(1., 1.), (2., 2.), (3., 3.), (5., 5.)]
    assert vs == [20., 15., 9., 7.]

    # duplicates + mean
    ks, vs = _combine_by_key(keys_with_duplicates, values_for_duplicates, 'mean')

    assert ks == [(1., 1.), (2., 2.), (3., 3.), (5., 5.)]
    assert vs == [16., 10., 6., 7.]