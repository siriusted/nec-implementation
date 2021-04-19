import torch
import numpy as np
from dnd import _combine_by_key, DND
from torch.nn import Parameter

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

def test_lookup():
    # test with initial setup
    config = { "capacity": 4, "neighbours": 2, "key_size": 2, "alpha": 0.5 }
    dnd = DND(config)

    key = torch.tensor([1., 2.], dtype = torch.float32, requires_grad = True)
    value = dnd.lookup(key)
    assert value == 0

    id_krnl = lambda dist: 1. / (dist + 1e-3)

    # test with exact match existing
    dnd.keys = Parameter(
        torch.tensor([
            [1., 2.],
            [1., 1.],
            [2., 1.],
            [3., 2.],
        ])
    )

    dnd.values = Parameter(
        torch.tensor([8., 2., 3., 5.])
    )

    value = dnd.lookup(key)
    weights = np.array([id_krnl(0), id_krnl(1.)])
    expected = ((weights / weights.sum()) * [8., 2.]).sum()
    assert np.isclose(value, expected)

    # test with no exact match
    dnd.keys = Parameter(torch.tensor([
        [3., 2.],
        [1., 1.],
        [2., 1.],
        [2., 2.],
    ]))

    value = dnd.lookup(key)
    weights = np.array([id_krnl(1)] * 2)
    expected = ((weights / weights.sum()) * [2., 5.]).sum()
    assert np.isclose(value, expected)

def test_forward():
    pass

def test_update_batch():
    pass