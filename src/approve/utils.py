from typing import Tuple, TypeAlias
from .typing import (
    EdgeTypeList,
    EdgeTypeFloatDict,
    EdgeTypeFloatOptDict,
    EdgeTypeTensorDict,
    EdgeTypeTensorOptDict,
    NodeTypeFloatDict,
    NodeTypeFloatOptDict,
    NodeTypeIntDict,
    NodeTypeIntOptDict,
    NodeTypeList,
    NodeTypeOptIntDict,
    NodeTypeOptIntOptDict,
    OptInt,
)
from torch_geometric.typing import OptTensor
import torch
from torch import Tensor
from torch_geometric.utils import (
    add_remaining_self_loops,
    scatter,
)
from torch_geometric.utils.num_nodes import (
    maybe_num_nodes,
    maybe_num_nodes_dict,
)
import numpy as np


pr_norm_type: TypeAlias = Tuple[Tensor, Tensor]
bipartite_pr_norm_type: TypeAlias = Tuple[
    Tensor, OptTensor, int, int, OptInt, OptInt
]
hetero_pr_norm_type: TypeAlias = Tuple[
    EdgeTypeTensorDict, EdgeTypeTensorDict, NodeTypeIntDict, NodeTypeOptIntDict
]


def missing_indices(
    index: Tensor,
    num_nodes: OptInt = None,
) -> OptTensor:
    r"""Finds all integers in the interval :math:`[0,N)` missing from
    :obj:`index`. If :obj:`num_nodes` is not :obj:`None`, :math:`N` is
    :obj:`num_nodes`. Otherwsie, :math:`N` is the maximum value of
    :obj:`index` plus one.

    Args:
        index (torch.Tensor): A one-dimensional tensor of non-negative integers
            representing nodes.
        num_nodes (int, optional): The number of nodes, if known.
            (default: :obj:`None`)

    Returns:
        A one-dimensional tensor of all missing indices or :obj:`None` if no
        indices are missing.

    Raises:
        AssertionError: If :obj:`num_nodes` is not :obj:`None` and not more
            than the maximum value of :obj:`index`.

    Examples:
        >>> index = torch.tensor([0, 2])
        >>> missing_index(index)
        tensor([1])
    """
    max_val = int(index.max().item())
    if num_nodes is None:
        num_nodes = max_val + 1
    else:
        assert (
            num_nodes > max_val
        ), "`num_nodes` must be larger than the maximum value of `index`."

    unique = index.unique()
    mask = index.new_full((num_nodes,), 1, dtype=torch.bool)
    mask[unique] = False

    if mask.any().item():
        missing = torch.arange(
            num_nodes, dtype=index.dtype, device=index.device
        )
        return missing[mask]

    return None


def bipartite_maybe_num_nodes(
    edge_index: Tensor,
    num_nodes_s: OptInt = None,
    num_nodes_t: OptInt = None,
) -> Tuple[int, int]:
    r"""Calculates the number of source and target nodes in the bipartite graph
    with edge indices given by :obj:`edge_index`.

    Args:
        edge_index (Tensor): The edge indices.
        num_nodes_s (int, optional): The number of source nodes, if known.
            (default: :obj:`None`)
        num_nodes_t (int, optional): The number of target nodes, if known.
            (default: :obj:`None`)

    Returns:
        The updated values for :obj:`num_nodes_s` and :obj:`num_nodes_t`.

    Raises:
        AssertionError: If :obj:`num_nodes_s` is not :obj:`None` and not more
            than the maximum value of :obj:`edge_index[0]`.
        AssertionError: If :obj:`num_nodes_t` is not :obj:`None` and not more
            than the maximum value of :obj:`edge_index[1]`.

    Shape:
        - Input:
          :math:`(2,|\mathcal{E}|)` where :math:`|\mathcal{E}|` is the number
          of edges.

    Examples:
        >>> edge_index = torch.tensor([[0, 0],
        ... [0, 1]])
        >>> bipartite_maybe_num_nodes(edge_index)
        (1, 2)
    """
    max_val_s = int(edge_index[0].max().item())
    if num_nodes_s is None:
        num_nodes_s = max_val_s + 1
    else:
        assert num_nodes_s > max_val_s, (
            "`num_nodes_s` must be larger than the maximum value of "
            "`edge_index[0]`."
        )

    max_val_t = int(edge_index[1].max().item())
    if num_nodes_t is None:
        num_nodes_t = max_val_t + 1
    else:
        assert num_nodes_t > max_val_t, (
            "`num_nodes_t` must be larger than the maximum value of "
            "`edge_index[1]`."
        )

    return num_nodes_s, num_nodes_t


def add_remaining_special_edges(
    edge_index: Tensor,
    edge_weight: OptTensor = None,
    num_nodes_s: OptInt = None,
    num_nodes_t: OptInt = None,
    special_s: OptInt = None,
    special_t: OptInt = None,
    fill_value: float = 1.0,
) -> Tuple[Tensor, OptTensor, OptInt, OptInt, OptInt, OptInt]:
    r"""Adds special edges, from missing source nodes to :obj:`special_t`,
    and from :obj:`special_s` to missing target nodes, for all nodes missing
    from :obj:`edge_index`. If :obj:`special_s` (resp. :obj:`special_t`) is not
    given, then it is set to :obj:`num_nodes_s` (resp. :obj:`num_nodes_t`).

    Args:
        edge_index (Tensor): The edge indices.
        edge_weight (Tensor, optional): The edge weights.
            (default: :obj:`None`)
        num_nodes_s (int, optional): The number of source nodes, if known.
            (default: :obj:`None`)
        num_nodes_t (int, optional): The number of target nodes, if known.
            (default: :obj:`None`)
        special_s (int, optional): The special source node, if set.
            (default: :obj:`None`)
        special_t (int, optional): The special target node, if set.
            (default: :obj:`None`)
        fill_value (float, optional): The weight associated with special edges.
            This is relevant only when :obj:`edge_weight` is not :obj:`None`.
            (default: :obj:`1.`)

    Returns:
        The updated values for :obj:`edge_weight`, :obj:`edge_weight`,
        :obj:`num_nodes_s`, :obj:`num_nodes_t`, :obj:`special_s`, and
        :obj:`special_t`.

    Shape:
        - Input:
          :math:`(2,|\mathcal{E}|)` and :math:`(|\mathcal{E}|)` where
          :math:`|\mathcal{E}|` is the number of edges.
        - Output:
          :math:`(2,|\mathcal{E}'|)` and :math:`(|\mathcal{E}'|)` where
          :math:`|\mathcal{E}'|` is the new number of edges.

    Examples:
        >>> edge_index = torch.tensor([[0, 0],
        ... [0, 1]])
        >>> add_remaining_special_edges(edge_index, num_nodes_s=2,
        ... num_nodes_t=3)
        (tensor([[0, 0, 1, 2],
                 [0, 1, 3, 2]]), None, 3, 4, 2, 3)
    """
    num_nodes_s, num_nodes_t = bipartite_maybe_num_nodes(
        edge_index,
        num_nodes_s,
        num_nodes_t,
    )

    device = edge_index.device

    dead_ends_s = missing_indices(edge_index[0], num_nodes_s)
    if dead_ends_s is not None:
        if special_t is None:
            special_t = num_nodes_t
            num_nodes_t += 1

        links_s = torch.cartesian_prod(
            dead_ends_s,
            torch.tensor([special_t], dtype=torch.long, device=device),
        ).t()

        edge_index = torch.cat((edge_index, links_s), dim=1).contiguous()
        if edge_weight is not None:
            edge_weight = torch.cat(
                (
                    edge_weight,
                    edge_weight.new_full((links_s.size(1),), fill_value),
                )
            )

    dead_ends_t = missing_indices(edge_index[1], num_nodes_t)
    if dead_ends_t is not None:
        if special_s is None:
            special_s = num_nodes_s
            num_nodes_s += 1

        links_t = torch.cartesian_prod(
            torch.tensor([special_s], dtype=torch.long, device=device),
            dead_ends_t,
        ).t()

        edge_index = torch.cat((edge_index, links_t), dim=1).contiguous()
        if edge_weight is not None:
            edge_weight = torch.cat(
                (
                    edge_weight,
                    edge_weight.new_full((links_t.size(1),), fill_value),
                )
            )

    return (
        edge_index,
        edge_weight,
        num_nodes_s,
        num_nodes_t,
        special_s,
        special_t,
    )


def _check_flow(flow: str) -> None:
    assert flow in [
        "source_to_target",
        "target_to_source",
    ], "`flow` must be either 'source_to_target' or 'target_to_source'"


def pr_norm(
    edge_index: Tensor,
    edge_weight: OptTensor = None,
    num_nodes: OptInt = None,
    add_self_loops: bool = True,
    flow: str = "source_to_target",
) -> pr_norm_type:
    r"""Calculates the entries of the (column) stochastic matrix
    :math:`\mathbf{\hat{S}} := \mathbf{\hat{A}}\mathbf{\hat{D}}^{-1}` as
    edge weights
    where
    :math:`\mathbf{\hat{A}}` is the adjacency matrix with self-loops
    possibly added, and
    :math:`\mathbf{\hat{D}}` is the diagonal degree matrix with entries
    :math:`\mathbf{\hat{D}}_{ii} := \sum_\ell \mathbf{\hat{A}}_{\ell i}`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_weight (Tensor, optional): The edge weights.
            (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, if known.
            (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`True`, the function
            will add self-loops to the graph.
            (default: :obj:`True`)
        flow (str, optional): The flow of the graph.
            (default: :obj:`"source_to_target"`)

    Returns:
        The updated values for :obj:`edge_index` and :obj:`edge_weight`.

    Raises:
        AssertionError: If :obj:`flow` is neither :obj:`"source_to_target"` nor
            :obj:`"target_to_source"`.

    Shape:
        - Input:
          :math:`(2,|\mathcal{E}|)` and :math:`(|\mathcal{E}|)` where
          :math:`|\mathcal{E}|` is the number of edges.
        - Output:
          :math:`(2,|\mathcal{E}'|)` and :math:`(|\mathcal{E}'|)` where
          :math:`|\mathcal{E}'|` is the new number of edges.
    """
    _check_flow(flow)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),),
            device=edge_index.device,
        )

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if add_self_loops:
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index,
            edge_attr=edge_weight,
            num_nodes=num_nodes,
            fill_value=1.0,
        )

    if flow == "source_to_target":
        idx = edge_index[0]
    else:
        idx = edge_index[1]

    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv = deg.float().pow_(-1)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if flow == "source_to_target":
        edge_weight = deg_inv[idx] * edge_weight
    else:
        edge_weight = edge_weight * deg_inv[idx]

    return edge_index, edge_weight  # type: ignore [return-value]


def bipartite_pr_norm(
    edge_index: Tensor,
    edge_weight: OptTensor = None,
    num_nodes_s: OptInt = None,
    num_nodes_t: OptInt = None,
    add_special_edges: bool = True,
    special_s: OptInt = None,
    special_t: OptInt = None,
    flow: str = "source_to_target",
) -> bipartite_pr_norm_type:
    r"""Calculates the entries of the (column) stochastic matrix
    :math:`\mathbf{\hat{S}} := \mathbf{\hat{A}}\mathbf{\hat{D}}^{-1}` as
    edge weights
    where
    :math:`\mathbf{\hat{A}}` is the adjacency matrix with special edges
    possibly added, and
    :math:`\mathbf{\hat{D}}` is the diagonal degree matrix with entries
    :math:`\mathbf{\hat{D}}_{ii} := \sum_\ell \mathbf{\hat{A}}_{\ell i}`.

    Args:
        edge_index (Tensor): The edge indices.
        edge_weight (Tensor, optional): The edge weights.
            (default: :obj:`None`)
        num_nodes_s (int, optional): The number of source nodes, if known.
            (default: :obj:`None`)
        num_nodes_t (int, optional): The number of target nodes, if known.
            (default: :obj:`None`)
        add_special_edges (bool, optional): If set to :obj:`True`, the function
            will add special edges to the bipartite graph.
            (default: :obj:`True`)
        special_s (int, optional): The special source node, if set.
            (default: :obj:`None`)
        special_t (int, optional): The special target node, if set.
            (default: :obj:`None`)
        flow (str, optional): The flow of the graph.
            (default: :obj:`"source_to_target"`)

    Returns:
        The updated values for :obj:`edge_index`, :obj:`edge_weight`,
        :obj:`num_nodes_s`, :obj:`num_nodes_t`, :obj:`special_s`, and
        :obj:`special_t`.

    Raises:
        AssertionError: If :obj:`flow` is neither :obj:`"source_to_target"` nor
            :obj:`"target_to_source"`.
    """
    _check_flow(flow)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),),
            device=edge_index.device,
        )

    num_nodes_s, num_nodes_t = bipartite_maybe_num_nodes(
        edge_index,
        num_nodes_s,
        num_nodes_t,
    )

    if add_special_edges:
        returned = add_remaining_special_edges(
            edge_index,
            edge_weight=edge_weight,
            num_nodes_s=num_nodes_s,
            num_nodes_t=num_nodes_t,
            special_s=special_s,
            special_t=special_t,
            fill_value=1.0,
        )
        edge_index = returned[0]
        edge_weight = returned[1]
        num_nodes_s = returned[2]
        num_nodes_t = returned[3]
        special_s = returned[4]
        special_t = returned[5]

    if flow == "source_to_target":
        idx = edge_index[0]
        num_nodes = num_nodes_s
    else:
        idx = edge_index[1]
        num_nodes = num_nodes_t

    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="sum")
    deg_inv = deg.float().pow_(-1)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    if flow == "source_to_target":
        edge_weight = deg_inv[idx] * edge_weight
    else:
        edge_weight = edge_weight * deg_inv[idx]

    return (
        edge_index,
        edge_weight,
        num_nodes_s,
        num_nodes_t,
        special_s,
        special_t,
    )  # type: ignore[return-value]


def hetero_pr_norm(
    edge_index_dict: EdgeTypeTensorDict,
    edge_weight_dict: EdgeTypeTensorOptDict = None,
    num_nodes_dict: NodeTypeIntOptDict = None,
    add_self_loops: bool = True,
    add_special_edges: bool = True,
    special_dict: NodeTypeOptIntOptDict = None,
    flow: str = "source_to_target",
) -> hetero_pr_norm_type:
    r"""Calculates the entries of the (column) stochastic matrix
    :math:`\mathbf{\hat{S}}[e] := \mathbf{\hat{A}}[e]\mathbf{\hat{D}}[e]^{-1}`
    as edge weights for all edge types :math:`e` where
    :math:`\mathbf{\hat{A}}[e]` is the adjacency matrix for type :math:`e`
    edges with self-loops (for homogenenous layers) or special edges
    (for bipartite layers) possibly added, and
    :math:`\mathbf{\hat{D}}[e]` is the diagonal degree matrix for type
    :math:`e` edges with entries
    :math:`\mathbf{\hat{D}}[e]_{ii} := \sum_\ell \mathbf{\hat{A}}[e]_{\ell i}`.

    Args:
        edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
            of edge indices for each edge type.
        edge_weight_dict (Dict[Tuple[str, str, str], Tensor], optional): A
            dictionary of edge weights for each edge type.
            (default: :obj:`None`)
        num_nodes_dict (Dict[str, Tensor], optional): A dictionary for the
            number of each node type, if known.
            (default: :obj:`None`)
        add_self_loops (bool, optional): If set to :obj:`True`, the function
            will add self-loops to all homogeneous layers.
            (default: :obj:`True`)
        add_special_edges (bool, optional): If set to :obj:`True`, the function
            will add special edges to all bipartite layers.
            (default: :obj:`True`)
        special_dict (Dict[str], Tensor], optional): A dictionary of special
            nodes for each node type, if set.
            (default: :obj:`None`)
        flow (str, optional): The flow of the graph.
            (default: :obj:`"source_to_target"`)

    Returns:
        The updated values for :obj:`edge_index_dict`, :obj:`edge_weight_dict`,
        :obj:`num_nodes_dict`, and :obj:`special_dict`.

    Raises:
        AssertionError: If :obj:`flow` is neither :obj:`"source_to_target"` nor
            :obj:`"target_to_source"`.
    """

    assert flow in ["source_to_target", "target_to_source", "undirected"]

    edge_weight_dict = edge_weight_dict or {}
    special_dict = special_dict or {}
    num_nodes_dict = maybe_num_nodes_dict(edge_index_dict, num_nodes_dict)

    # Iterate over bipartite graphs
    _num_nodes_dict: NodeTypeIntDict = {}
    _special_dict: NodeTypeOptIntDict = {}
    for edge_type, edge_index in edge_index_dict.items():
        if edge_type[0] != edge_type[-1]:
            num_nodes_s = num_nodes_dict.get(edge_type[0])  # type: ignore[union-attr]
            num_nodes_t = num_nodes_dict.get(edge_type[-1])  # type: ignore[union-attr]
            bipartite_norm: bipartite_pr_norm_type = bipartite_pr_norm(
                edge_index,
                edge_weight=edge_weight_dict.get(edge_type),
                num_nodes_s=num_nodes_s,
                num_nodes_t=num_nodes_t,
                add_special_edges=add_special_edges,
                special_s=special_dict.get(edge_type[0]),
                special_t=special_dict.get(edge_type[-1]),
                flow=flow,
            )
            edge_index_dict[edge_type] = bipartite_norm[0]
            edge_weight_dict[edge_type] = bipartite_norm[1]
            _num_nodes_dict[edge_type[0]] = bipartite_norm[2]
            _num_nodes_dict[edge_type[-1]] = bipartite_norm[3]
            _special_dict[edge_type[0]] = bipartite_norm[4]
            _special_dict[edge_type[-1]] = bipartite_norm[5]
    num_nodes_dict = _num_nodes_dict
    special_dict = _special_dict

    # Iterate over homogeneous graphs
    for edge_type, edge_index in edge_index_dict.items():
        if edge_type[0] == edge_type[-1]:
            num_nodes = num_nodes_dict.get(edge_type[0])  # type: ignore [union-attr]
            norm: pr_norm_type = pr_norm(
                edge_index,
                edge_weight=edge_weight_dict.get(edge_type),
                num_nodes=num_nodes,
                add_self_loops=add_self_loops,
                flow=flow,
            )
            edge_index_dict[edge_type] = norm[0]
            edge_weight_dict[edge_type] = norm[1]

    return (  # type: ignore[return-value]
        edge_index_dict,
        edge_weight_dict,
        num_nodes_dict,
        special_dict,
    )


def gen_alpha_dict(
    node_type_list: NodeTypeList,
    alpha_exp_dict: NodeTypeFloatOptDict = None,
) -> NodeTypeFloatDict:
    r"""Calculates the transport probabilities :math:`\alpha[n]` for each
    node type :math:`n` in :obj:`node_type_list` via

    .. math::
        \alpha[n] = \frac{\exp{A[n]}}{1 + \exp{A[n]}}\,.

    Note:
        If the exponent :math:`A[n]` is not specified in
        :obj:`alpha_exp_dict`, then it defaults to :obj:`0`.

    Args:
        node_type_list (List[str]): A list of all node types.
        alpha_exp_dict (Dict[str, float], optional): A dictionary of exponents
            :math:`A[n]` used to calculate the transport probabilities
            :math:`\alpha[n]` for each node type.

    Returns:
        A dictionary of transport probabilities :math:`\alpha[n]` for each
        node type.
    """
    alpha_dict: NodeTypeFloatDict = {}
    alpha_exp_dict = alpha_exp_dict or {}

    for node_type in node_type_list:
        alpha: float = np.exp(alpha_exp_dict.get(node_type, 0))
        alpha = alpha / (alpha + 1)
        alpha_dict[node_type] = alpha

    return alpha_dict


def gen_beta_dict(
    edge_type_list: EdgeTypeList,
    beta_exp_dict: EdgeTypeFloatOptDict = None,
) -> EdgeTypeFloatDict:
    r"""Calculates the contribution percentages :math:`\beta[e]` for
    each edge type :math:`e:n' \to n` in :obj:`beta_exp_list` via

    .. math::
        \beta[e] =
        \frac{\exp{B[e]}}{\sum_{n'' \in \mathcal{N}}
        \sum_{e'\in\mathcal{E}[n'',n]} \exp{B[e']}}\,,

    where the sum is over all edge types with type :math:`n` nodes as targets.

    Note:
        If the exponent :math:`B[e]` is not specified in :obj:`beta_exp_dict`,
        then it defaults to :obj:`0`.

    Args:
        edge_type_list (List[Tuple[str, str, str]]): A list of all edge types.
        beta_exp_dict (Dict[Tuple[str, str, str], float], optional): A
            dictionary of exponents :math:`B[e]` used to calculate the
            contribution percentages :math:`\beta[e]` for each edge type.

    Returns:
        A dictionary of contribution percentages :math:`\beta[e]` for each edge
        type.
    """
    beta_dict: EdgeTypeFloatDict = {}
    beta_exp_dict = beta_exp_dict or {}
    total: NodeTypeFloatDict = {}

    for edge_type in edge_type_list:
        beta: float = np.exp(beta_exp_dict.get(edge_type, 0))
        beta_dict[edge_type] = beta
        total[edge_type[-1]] = total.get(edge_type[-1], 0) + beta

    for edge_type, beta in beta_dict.items():
        beta_dict[edge_type] = beta / total[edge_type[-1]]

    return beta_dict


def _check_node_type_conn(
    node_type_list: NodeTypeList,
    edge_type_list: EdgeTypeList,
) -> None:
    targets = set([edge_type[-1] for edge_type in edge_type_list])
    node_type_set = set(node_type_list)

    assert (
        node_type_set <= targets
    ), "Every node type must be the target of at least one edge type."
