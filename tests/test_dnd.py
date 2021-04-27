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

id_krnl = lambda dist: 1. / (dist + 1e-3)
config = { "dnd_capacity": 4, "num_neighbours": 2, "key_size": 2, "alpha": 0.5 }

def test_lookup():
    # test with initial setup
    dnd = DND(config)

    key = torch.tensor([1., 2.], dtype = torch.float32, requires_grad = True)
    value = dnd.lookup(key)
    assert np.array_equal(dnd.last_used, np.array([0, 0, 3, 2]))
    assert value == 0

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
    assert np.array_equal(dnd.last_used, np.array([0, 0, 4, 3]))
    assert np.isclose(value, expected, atol=1e-3)

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
    assert np.array_equal(dnd.last_used, np.array([1, 0, 5, 0]))
    assert np.isclose(value, expected, atol=1e-3)

def test_forward():
    dnd = DND(config)

    keys = torch.tensor([
        [1., 2.],
        [3., 4.]
    ], requires_grad = True)

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

    values = dnd(keys)
    weights = np.array([
        [id_krnl(0), id_krnl(1.)],
        [id_krnl(2.), id_krnl(np.sqrt(8.))]
    ])
    expected = (weights / weights.sum(axis = 1).reshape(2, 1) * np.array([[8., 2.], [5., 8.]])).sum(axis = 1)

    assert values.grad_fn is not None
    assert np.array_equal(dnd.last_used, np.array([0, 0, 3, 0]))
    assert np.allclose(values.detach().numpy(), expected, atol=1e-3)


def test_update_batch():
    # test initial update
    dnd = DND(config)

    keys = torch.stack([
        torch.tensor([1., 1.]),
        torch.tensor([2., 2.]),
    ])

    values = np.array([1., 2.], dtype = np.float32)

    # reset last_used for easier testing
    dnd.last_used = np.zeros(dnd.capacity, dtype = np.uint32)
    dnd.update_batch(keys, values)
    expected_keys_hash = {
        (1., 1.): 2,
        (2., 2.): 3
    }

    assert np.array_equal(dnd.last_used, np.array([1, 1, 0, 0]))
    assert dnd.keys_hash == expected_keys_hash


    # test update with mix of existing keys and new keys
    keys = torch.stack([
        torch.tensor([1., 1.]),
        torch.tensor([2., 2.]),
        torch.tensor([3., 2.]),
    ])

    values = np.array([3., 8., 1.], dtype = np.float32)

    dnd.update_batch(keys, values)
    expected_keys_hash = {
        (3., 2.): 1,
        (1., 1.): 2,
        (2., 2.): 3,
    }
    expected_values = np.array([0., 1., 2., 5.], dtype = np.float32)

    assert np.array_equal(dnd.last_used, np.array([2, 0, 0, 0]))
    assert dnd.keys_hash == expected_keys_hash
    assert np.allclose(dnd.values.detach().numpy(),  expected_values)
