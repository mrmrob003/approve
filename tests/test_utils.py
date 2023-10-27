import numpy as np
import pytest
import torch

from approve.utils import (
    _check_flow,
    _check_node_type_conn,
    add_remaining_special_edges,
    bipartite_maybe_num_nodes,
    bipartite_pr_norm,
    gen_alpha_dict,
    gen_beta_dict,
    hetero_pr_norm,
    missing_indices,
    pr_norm,
)

# =============================
# === Test: missing_indices ===
# =============================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(), torch.tensor([1])),
        (dict(num_nodes=4), torch.tensor([1, 3])),
    ],
)
def test_missing_indices(kwargs, expected):
    index = torch.tensor([0, 2, 0])
    assert torch.equal(missing_indices(index, **kwargs), expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(num_nodes=2),
    ],
)
def test_missing_indices_none(kwargs):
    index = torch.tensor([0, 1, 0])
    assert missing_indices(index, **kwargs) is None


def test_missing_indices_assert():
    index = torch.tensor([0])
    with pytest.raises(AssertionError):
        missing_indices(index, 0)


# =======================================
# === Test: bipartite_maybe_num_nodes ===
# =======================================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(), (1, 2)),
        (dict(num_nodes_s=2), (2, 2)),
        (dict(num_nodes_t=3), (1, 3)),
    ],
)
def test_bipartite_maybe_num_nodes(kwargs, expected):
    edge_index = torch.tensor([[0, 0], [0, 1]])
    returned = bipartite_maybe_num_nodes(edge_index, **kwargs)
    assert returned == expected


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_nodes_s=0),
        dict(num_nodes_t=1),
    ],
)
def test_bipartite_maybe_num_nodes_assert(kwargs):
    edge_index = torch.tensor([[0, 0], [0, 1]])
    with pytest.raises(AssertionError):
        bipartite_maybe_num_nodes(edge_index, **kwargs)
        bipartite_maybe_num_nodes(edge_index, 0, 1)


# =========================================
# === Test: add_remaining_special_edges ===
# =========================================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(), (torch.tensor([[0, 0], [0, 1]]), None, 1, 2, None, None)),
        (
            dict(edge_weight=torch.tensor([1, 1])),
            (
                torch.tensor([[0, 0], [0, 1]]),
                torch.tensor([1, 1]),
                1,
                2,
                None,
                None,
            ),
        ),
        (
            dict(num_nodes_s=2),
            (torch.tensor([[0, 0, 1], [0, 1, 2]]), None, 2, 3, None, 2),
        ),
        (
            dict(edge_weight=torch.tensor([1, 1]), num_nodes_s=2),
            (
                torch.tensor([[0, 0, 1], [0, 1, 2]]),
                torch.tensor([1, 1, 1]),
                2,
                3,
                None,
                2,
            ),
        ),
        (
            dict(num_nodes_s=2, special_t=0),
            (torch.tensor([[0, 0, 1], [0, 1, 0]]), None, 2, 2, None, 0),
        ),
        (
            dict(num_nodes_t=3),
            (torch.tensor([[0, 0, 1], [0, 1, 2]]), None, 2, 3, 1, None),
        ),
        (
            dict(edge_weight=torch.tensor([1, 1]), num_nodes_t=3),
            (
                torch.tensor([[0, 0, 1], [0, 1, 2]]),
                torch.tensor([1, 1, 1]),
                2,
                3,
                1,
                None,
            ),
        ),
        (
            dict(num_nodes_t=3, special_s=0),
            (torch.tensor([[0, 0, 0], [0, 1, 2]]), None, 1, 3, 0, None),
        ),
        (
            dict(
                edge_weight=torch.tensor([1, 1]),
                num_nodes_s=2,
                num_nodes_t=3,
                fill_value=2,
            ),
            (
                torch.tensor([[0, 0, 1, 2], [0, 1, 3, 2]]),
                torch.tensor([1, 1, 2, 2]),
                3,
                4,
                2,
                3,
            ),
        ),
    ],
)
def test_add_remaining_special_edges(kwargs, expected):
    edge_index = torch.tensor([[0, 0], [0, 1]])
    returned = add_remaining_special_edges(edge_index, **kwargs)
    assert torch.equal(returned[0], expected[0])
    if expected[1] is None:
        assert returned[1] is None
    else:
        assert torch.equal(returned[1], expected[1])
    assert returned[2] == expected[2]
    assert returned[3] == expected[3]
    assert returned[4] == expected[4]
    assert returned[5] == expected[5]


# ============================
# === Test: _check_flow ===
# ============================


@pytest.mark.parametrize("flow", ["source_to_target", "target_to_source"])
def test__check_flow(flow):
    assert _check_flow(flow) is None


def test__check_flow_assert():
    with pytest.raises(AssertionError):
        _check_flow("")


# =====================
# === Test: pr_norm ===
# =====================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(),
            (
                torch.tensor([[0, 0, 1], [1, 0, 1]]),
                torch.tensor([0.5, 0.5, 1]),
            ),
        ),
        (
            dict(edge_weight=torch.tensor([2])),
            (
                torch.tensor([[0, 0, 1], [1, 0, 1]]),
                torch.tensor([2 / 3, 1 / 3, 1]),
            ),
        ),
        (
            dict(num_nodes=3),
            (
                torch.tensor([[0, 0, 1, 2], [1, 0, 1, 2]]),
                torch.tensor([0.5, 0.5, 1, 1]),
            ),
        ),
        (
            dict(add_self_loops=False),
            (torch.tensor([[0], [1]]), torch.tensor([1.0])),
        ),
        (
            dict(flow="target_to_source"),
            (
                torch.tensor([[0, 0, 1], [1, 0, 1]]),
                torch.tensor([0.5, 1, 0.5]),
            ),
        ),
    ],
)
def test_pr_norm(kwargs, expected):
    edge_index = torch.tensor([[0], [1]])
    returned = pr_norm(edge_index, **kwargs)
    assert torch.equal(returned[0], expected[0])
    assert torch.allclose(returned[1], expected[1])


# ===============================
# === Test: bipartite_pr_norm ===
# ===============================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(),
            (
                torch.tensor([[0, 0], [0, 1]]),
                torch.tensor([0.5, 0.5]),
                1,
                2,
                None,
                None,
            ),
        ),
        (
            dict(edge_weight=torch.tensor([2, 1])),
            (
                torch.tensor([[0, 0], [0, 1]]),
                torch.tensor([2 / 3, 1 / 3]),
                1,
                2,
                None,
                None,
            ),
        ),
        (
            dict(num_nodes_s=2),
            (
                torch.tensor([[0, 0, 1], [0, 1, 2]]),
                torch.tensor([0.5, 0.5, 1]),
                2,
                3,
                None,
                2,
            ),
        ),
        (
            dict(num_nodes_s=2, special_t=0),
            (
                torch.tensor([[0, 0, 1], [0, 1, 0]]),
                torch.tensor([0.5, 0.5, 1]),
                2,
                2,
                None,
                0,
            ),
        ),
        (
            dict(num_nodes_t=3),
            (
                torch.tensor([[0, 0, 1], [0, 1, 2]]),
                torch.tensor([0.5, 0.5, 1]),
                2,
                3,
                1,
                None,
            ),
        ),
        (
            dict(num_nodes_t=3, special_s=0),
            (
                torch.tensor([[0, 0, 0], [0, 1, 2]]),
                torch.tensor([1 / 3, 1 / 3, 1 / 3]),
                1,
                3,
                0,
                None,
            ),
        ),
        (
            dict(num_nodes_s=2, num_nodes_t=3, add_special_edges=False),
            (
                torch.tensor([[0, 0], [0, 1]]),
                torch.tensor([0.5, 0.5]),
                2,
                3,
                None,
                None,
            ),
        ),
        (
            dict(flow="target_to_source"),
            (
                torch.tensor([[0, 0], [0, 1]]),
                torch.tensor([1.0, 1.0]),
                1,
                2,
                None,
                None,
            ),
        ),
    ],
)
def test_bipartite_pr_norm(kwargs, expected):
    edge_index = torch.tensor([[0, 0], [0, 1]])
    returned = bipartite_pr_norm(edge_index, **kwargs)
    assert torch.equal(returned[0], expected[0])
    assert torch.allclose(returned[1], expected[1])
    assert returned[2] == expected[2]
    assert returned[3] == expected[3]
    assert returned[4] == expected[4]
    assert returned[5] == expected[5]


# ============================
# === Test: hetero_pr_norm ===
# ============================


def test_hetero_pr_norm(ex_hetero_data, ex_hetero_data_normalized):
    edge_index_dict = ex_hetero_data.edge_index_dict
    returned = hetero_pr_norm(edge_index_dict)

    # check edge_index
    edge_index_dict = ex_hetero_data_normalized.edge_index_dict
    for edge_type, edge_index in returned[0].items():
        assert torch.equal(edge_index_dict[edge_type], edge_index)

    # check edge_weight
    edge_weight_dict = ex_hetero_data_normalized.edge_weight_dict
    for edge_type, edge_weight in returned[1].items():
        assert torch.allclose(edge_weight_dict[edge_type], edge_weight)

    # check num_nodes
    num_nodes_dict = ex_hetero_data_normalized.num_nodes_dict
    for node_type, num_nodes in returned[2].items():
        assert num_nodes_dict[node_type] == num_nodes

    # check special
    special_dict = ex_hetero_data_normalized.special_dict
    for node_type, special in returned[3].items():
        assert special_dict.get(node_type) == special


# ============================
# === Test: gen_alpha_dict ===
# ============================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (dict(), {"A": 0.5, "B": 0.5}),
        (
            dict(alpha_exp_dict={"A": 1}),
            {"A": np.exp(1) / (1 + np.exp(1)), "B": 0.5},
        ),
    ],
)
def test_alpha_dict(kwargs, expected):
    node_type_list = ["A", "B"]
    returned = gen_alpha_dict(node_type_list, **kwargs)
    for node_type, alpha in returned.items():
        assert np.isclose(expected[node_type], alpha)


# ===========================
# === Test: gen_beta_dict ===
# ===========================


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            dict(),
            {
                ("A", "to", "A"): 0.5,
                ("A", "to", "B"): 1,
                ("B", "to", "A"): 0.5,
            },
        ),
        (
            dict(beta_exp_dict={("A", "to", "A"): 1}),
            {
                ("A", "to", "A"): np.exp(1) / (1 + np.exp(1)),
                ("A", "to", "B"): 1,
                ("B", "to", "A"): 1 / (1 + np.exp(1)),
            },
        ),
    ],
)
def test_edge_frac_dict(kwargs, expected):
    edge_type_list = [("A", "to", "A"), ("A", "to", "B"), ("B", "to", "A")]
    returned = gen_beta_dict(edge_type_list, **kwargs)
    for edge_type, beta in returned.items():
        assert np.isclose(expected[edge_type], beta)


# ===================================
# === Test: _check_node_type_conn ===
# ===================================


def test__check_node_type_conn():
    node_type_list = ["A", "B"]
    edge_type_list = [("A", "to", "A"), ("A", "to", "B")]
    returned = _check_node_type_conn(node_type_list, edge_type_list)
    assert returned is None


def test__check_node_type_connected_assert():
    node_type_list = ["A", "B"]
    edge_type_list = [
        ("A", "to", "B"),
    ]
    with pytest.raises(AssertionError):
        _check_node_type_conn(node_type_list, edge_type_list)
