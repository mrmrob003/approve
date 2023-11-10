import pytest
import torch

from approve.models import APPr, HeteroAPPr


def test_APPr():
    x = torch.tensor([[0.5], [0.5]])
    edge_index = torch.tensor([[0], [1]])

    model = APPr(K=30, alpha=0.5)
    assert str(model) == "APPr(K=30, alpha=0.5)"

    def check_returned():
        returned = model(x, edge_index)
        assert torch.allclose(returned, torch.tensor([[1 / 3], [2 / 3]]))

    check_returned()
    assert model._norm is not None
    check_returned()

    model.reset_parameters()
    assert model._norm is None


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),
        dict(alpha_exp_dict={"paper": 0, "venue": 0}),
        dict(
            beta_exp_dict={
                ("paper", "cites", "paper"): 0,
                ("venue", "publishes", "paper"): 0,
                ("paper", "rev_publishes", "venue"): 0,
            }
        ),
    ],
)
def test_HeteroAPPr(ex_hetero_data, kwargs):
    x_dict = {}
    for node_type, num_nodes in ex_hetero_data.num_nodes_dict.items():
        x_dict[node_type] = torch.ones((num_nodes, 1)) / num_nodes
    edge_index_dict = ex_hetero_data.edge_index_dict

    model = HeteroAPPr(K=30)
    assert str(model) == "HeteroAPPr(K=30)"

    def check_returned():
        returned = model(x_dict, edge_index_dict, **kwargs)
        assert torch.allclose(
            returned["paper"],
            torch.tensor([[0.4511278272], [0.3383458555], [0.2105263323]]),
        )
        assert torch.allclose(
            returned["venue"], torch.tensor([[0.8947368264], [0.1052631661]])
        )

    check_returned()
    assert model._norm is not None
    assert model._alpha_dict is not None
    assert model._beta_dict is not None
    check_returned()

    model.reset_parameters()
    assert model._norm is None
    assert model._alpha_dict is None
    assert model._beta_dict is None
