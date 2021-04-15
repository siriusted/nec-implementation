import torch
from dnd import _combine_by_key

def test_combine_by_key():
    # Test with no duplicates + max
    keys = [
        torch.tensor([1., 1.]),
        torch.tensor([2., 2.]),
        torch.tensor([3., 3.])
    ]

    values = [1., 2., 3.]

    ks, vs = _combine_by_key(keys, values, 'max')

    assert ks == [(1., 1.), (2., 2.), (3., 3.)]
    assert vs == values



