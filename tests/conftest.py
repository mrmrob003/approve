import pytest
import torch
from torch_geometric.data import HeteroData


@pytest.fixture
def ex_hetero_data():
    hetero_data = HeteroData()

    # edge_index_dict
    hetero_data["paper", "cites", "paper"].edge_index = torch.tensor(
        [[1, 2, 2], [0, 0, 1]]
    )
    hetero_data["venue", "publishes", "paper"].edge_index = torch.tensor(
        [[0, 0], [0, 1]]
    )
    hetero_data["paper", "rev_publishes", "venue"].edge_index = hetero_data[
        "venue", "publishes", "paper"
    ].edge_index[[1, 0]]

    # num_nodes_dict
    hetero_data["paper"].num_nodes = 3
    hetero_data["venue"].num_nodes = 1

    return hetero_data


@pytest.fixture
def ex_hetero_data_normalized():
    hetero_data = HeteroData()

    # edge_index_dict
    hetero_data["paper", "cites", "paper"].edge_index = torch.tensor(
        [[1, 2, 2, 0, 1, 2], [0, 0, 1, 0, 1, 2]]
    )
    hetero_data["venue", "publishes", "paper"].edge_index = torch.tensor(
        [[0, 0, 1], [0, 1, 2]]
    )
    hetero_data["paper", "rev_publishes", "venue"].edge_index = hetero_data[
        "venue", "publishes", "paper"
    ].edge_index[[1, 0]]

    # edge_weight_dict
    hetero_data["paper", "cites", "paper"].edge_weight = torch.tensor(
        [0.5, 1 / 3, 1 / 3, 1.0, 0.5, 1 / 3]
    )
    hetero_data["venue", "publishes", "paper"].edge_weight = torch.tensor(
        [0.5, 0.5, 1.0]
    )
    hetero_data["paper", "rev_publishes", "venue"].edge_weight = torch.tensor(
        [1.0, 1.0, 1.0]
    )

    # num_nodes_dict
    hetero_data["paper"].num_nodes = 3
    hetero_data["venue"].num_nodes = 2

    # special_dict
    hetero_data["paper"].special = None
    hetero_data["venue"].special = 1

    return hetero_data
