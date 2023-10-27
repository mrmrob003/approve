from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptTensor

from .typing import (
    EdgeTypeFloatDict,
    EdgeTypeFloatOptDict,
    EdgeTypeTensorDict,
    EdgeTypeTensorOptDict,
    NodeTypeFloatDict,
    NodeTypeFloatOptDict,
    NodeTypeOptIntOptDict,
    NodeTypeTensorDict,
)
from .utils import (
    _check_node_type_conn,
    gen_alpha_dict,
    gen_beta_dict,
    hetero_pr_norm,
    hetero_pr_norm_type,
    pr_norm,
    pr_norm_type,
)


class APPr(MessagePassing):
    r"""The approximate personalized PageRank model for homogeneous graphs,
    adapted from the `APPNP model <https://pytorch-geometric.readthedocs.io/en/
    latest/generated/torch_geometric.nn.conv.APPNP.html>`_, as detailed in the
    paper `"APPrOVE: Approximate Personalized Propagation Over Varied Edges"
    <https://arxiv.org/abs/23xx.xxxxx>`_:

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}\,,

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{S}} \mathbf{X}^{(k-1)}
        + \alpha \mathbf{X}^{(0)}\,,

        \mathbf{X}' &= \mathbf{X}^{(K)}\,,

    where
    :math:`\alpha` is the transport probability,
    :math:`\mathbf{\hat{S}}` is the (column) stochastic matrix
    :math:`\mathbf{\hat{S}} := \mathbf{\hat{A}}\mathbf{\hat{D}}^{-1}`,
    :math:`\mathbf{\hat{A}}` is the adjacency matrix with self-loops
    possibly added, and
    :math:`\mathbf{\hat{D}}` is the diagonal degree matrix with entries
    :math:`\mathbf{\hat{D}}_{ii} := \sum_\ell \mathbf{\hat{A}}_{\ell i}`.

    Note:
        The adjacency matrix can include values different from :obj:`1`
        (representing edge weights) via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): The number of iterations :math:`K`.
        alpha (float): The teleport probability :math:`\alpha`.
        add_self_loops (bool, optional): If set to :obj:`True`, the model will
            add self-loops to the graph.
            (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the model will cache
            the computation of :math:`\mathbf{\hat{A}} \mathbf{\hat{D}}` on the
            first execution and use the cached version for further executions.
            (default: :obj:`True`)
        normalize (bool, optional): If set to :obj:`True`, the model will add
            self-loops to the graph and normalize the edge weights.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Examples:
        .. code-block:: python

            from approve.models import APPr
            from torch import tensor

            # define homogeneous graph
            edge_index = torch.tensor([[1, 2, 2],
                                    [0, 0, 1]])

            # assign uniform scores to each node
            x = torch.full((3, 1), 1 / 3)

            # compute PageRank
            model = APPr(K=30, alpha=0.5)
            model(x, edge_index)
            >>> tensor([[0.5333],
                        [0.2667],
                        [0.2000]])
    """

    def __init__(
        self,
        K: int,
        alpha: float,
        add_self_loops: bool = True,
        cached: bool = True,
        normalize: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.K = K
        self.alpha = alpha
        self.add_self_loops = add_self_loops
        self.cached = cached
        self.normalize = normalize

        self._norm: Optional[pr_norm_type] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self._norm = None

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_weight: OptTensor = None
    ) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (Tensor): The node features :math:`\mathbf{X}`.
            edge_index (Tensor): The edge indices.
            edge_weight (Tensor, optional): The edge weights.
                (default: :obj:`None`)

        Returns:
            The updated node features :math:`\mathbf{X}'`.

        Shape:
            - Input:
              :math:`(|\mathcal{N}|, F)`, :math:`(2,|\mathcal{E}|)`, and
              :math:`(|\mathcal{E}|)`, where :math:`|\mathcal{N}|` is the
              number of nodes, :math:`|\mathcal{E}|` is the number of edges,
              and :math:`F` is the number of features.
            - Output: :math:`(|\mathcal{N}|, F)`.
        """
        # compute and cache norm
        if self.normalize:
            norm: Optional[pr_norm_type] = self._norm
            if norm is None:
                norm = pr_norm(
                    edge_index,
                    edge_weight=edge_weight,
                    add_self_loops=self.add_self_loops,
                )
                if self.cached:
                    self._norm = norm
            edge_index, edge_weight = norm

        h = x
        for _ in range(self.K):
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight)
            x = x * (1 - self.alpha)
            x = x + self.alpha * h

        return x

    @staticmethod
    def message(x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K}, alpha={self.alpha})"


class APPrOVE(MessagePassing):
    r"""The approximate personalized PageRank model for heterogeneous graphs,
    as detailed in the paper `"APPrOVE: Approximate Personalized Propagation
    Over Varied Edges" <https://arxiv.org/abs/23xx.xxxxx>`_:

    .. math::
        \mathbf{X}^{[n](0)} &= \mathbf{X}^{[n]}\,,

        \mathbf{X}^{[n](k+1)} &= (1 - \alpha[n]) \textstyle
        \sum_{n' \in \mathcal{N}} \sum_{e \in \mathcal{E}[n',n]}
        \beta[e] \mathbf{\hat{S}}[e] \mathbf{X}^{[n'](k)}
        + \alpha[n] \mathbf{X}^{[n](0)}\,,

        \mathbf{X}^{\prime[n]} &= \mathbf{X}^{[n](K)}\,,

    where
    :math:`\alpha[n]` is the transport probability for type :math:`n`
    nodes,
    :math:`\mathcal{N}` is the set of all node types,
    :math:`\mathcal{E}[n',n]` is the set of all edge types :math:`e: n' \to n`,
    :math:`\beta[e]` is the contribution percentage from type :math:`e`
    edges for updating the features of type :math:`n` nodes,
    :math:`\mathbf{\hat{S}}[e]` is the (column) stochastic matrix
    :math:`\mathbf{\hat{S}}[e] := \mathbf{\hat{A}}[e]\mathbf{\hat{D}}[e]^{-1}`,
    :math:`\mathbf{\hat{A}}[e]` is the adjacency matrix for type :math:`e`
    edges with self-loops (for homogenenous layers) or special edges
    (for bipartite layers) possibly added, and
    :math:`\mathbf{\hat{D}}[e]` is the diagonal degree matrix for type
    :math:`e` edges with entries
    :math:`\mathbf{\hat{D}}[e]_{ii} := \sum_\ell \mathbf{\hat{A}}[e]_{\ell i}`.

    Note:
        All adjacency matrices can include values other than :obj:`1`
        (representing edge weights) via the optional :obj:`edge_weight_dict`
        dictionary.

    Args:
        K (int): The number of iterations :math:`K`.
        add_self_loops (bool, optional): If set to :obj:`True`, the model will
            add self-loops to all homogeneous layers.
            (default: :obj:`True`)
        add_special_edges (bool, optional): If set to :obj:`True`, the model
            will add special edges to all bipartite layers.
            (default: :obj:`True`)
        cached (bool, optional): If set to :obj:`True`, the model will cache
            the computation of :math:`\mathbf{\hat{S}}[e]` on the first
            execution and use the cached version for further executions.
            (default: :obj:`True`)
        normalize (bool, optional): If set to :obj:`True`, the model will add
            self-loops to homogeneous layers and special edges to bipartite
            layers, and normalize the edge weights.
            (default: :obj:`True`)
        check_node_type_conn (bool, optional): If set to :obj:`True`, the model
            checks that every node type is the target of at least one edge
            type.
            (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`

    Examples:
        .. code-block:: python

            from approve.models import APPrOVE
            from torch import tensor
            from torch_geometric.data import HeteroData

            # define heterogeneous graph
            hetero_data = HeteroData()
            hetero_data['paper', 'cites', 'paper'].edge_index = \
                torch.tensor([[1, 2, 2],
                              [0, 0, 1]])
            hetero_data['venue', 'publishes', 'paper'].edge_index = \
                torch.tensor([[0, 1],
                              [0, 1]])
            hetero_data['paper', 'rev_publishes', 'venue'].edge_index = \
                hetero_data['venue', 'publishes', 'paper'].edge_index[[1,0]]

            # assign uniform scores to each node
            hetero_data['paper'].x = torch.full((3, 1), 1 / 3)
            hetero_data['venue'].x = torch.full((2, 1), 1 / 2)

            # compute PageRank
            model = APPrOVE(K=30)
            model(hetero_data.x_dict, hetero_data.edge_index_dict)
            >>> {'paper': tensor([[0.4605],
                         [0.3289],
                         [0.2105]]),
                 'venue': tensor([[0.4803],
                         [0.4145],
                         [0.1053]])}
    """

    def __init__(
        self,
        K: int,
        add_self_loops: bool = True,
        add_special_edges: bool = True,
        cached: bool = True,
        normalize: bool = True,
        check_node_type_conn: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.K = K
        self.add_self_loops = add_self_loops
        self.add_special_edges = add_special_edges
        self.cached = cached
        self.normalize = normalize
        self.check_node_type_conn = check_node_type_conn

        self._norm: Optional[hetero_pr_norm_type] = None
        self._alpha_dict: Optional[NodeTypeFloatDict] = None
        self._beta_dict: Optional[EdgeTypeFloatDict] = None

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self._norm = None
        self._alpha_dict = None
        self._beta_dict = None

    def forward(
        self,
        x_dict: NodeTypeTensorDict,
        edge_index_dict: EdgeTypeTensorDict,
        edge_weight_dict: EdgeTypeTensorOptDict = None,
        special_dict: NodeTypeOptIntOptDict = None,
        alpha_exp_dict: NodeTypeFloatOptDict = None,
        beta_exp_dict: EdgeTypeFloatOptDict = None,
    ) -> NodeTypeTensorDict:
        r"""Runs the forward pass of the module.

        Args:
            x_dict (Dict[str, Tensor]): A dictionary of features for each
                node type.
            edge_index_dict (Dict[Tuple[str, str, str], Tensor]): A dictionary
                of edge indices for each edge type.
            edge_weight_dict (Dict[Tuple[str, str, str], Tensor], optional): A
                dictionary of edge weights for each edge type.
                (default: :obj:`None`)
            special_dict (Dict[str, int] , optional): A dictionary of special
                nodes for each node type.
                (default: :obj:`None`)
            alpha_exp_dict (Dict[str, float], optional): A dictionary of
                exponents used to calculate transport probabilities via
                :obj:`utils.gen_alpha_dict` for each node type.
                (default: :obj:`None`)
            beta_exp_dict (Dict[Tuple[str, str, str], float], optional): A
                dictionary of exponents used to calculate contribution
                percentages via :obj:`utils.gen_beta_dict` for each edge type.
                (default: :obj:`None`)

        Returns:
            An updated dictionary of features for each node type.

        Raises:
            AssertionError: If :obj:`self.check_node_type_conn` is set to
                :obj:`True` and not every node type is the target of at least
                one edge type.
        """
        # check node type connectivity
        if self.check_node_type_conn:
            node_type_list = list(x_dict.keys())
            edge_type_list = list(edge_index_dict.keys())
            _check_node_type_conn(node_type_list, edge_type_list)

        # compute and cache norm
        if self.normalize:
            norm: Optional[hetero_pr_norm_type] = self._norm
            if norm is None:
                norm = hetero_pr_norm(
                    edge_index_dict,
                    edge_weight_dict=edge_weight_dict,
                    add_self_loops=self.add_self_loops,
                    add_special_edges=self.add_special_edges,
                    special_dict=special_dict,
                )
                if self.cached:
                    self._norm = norm
            edge_index_dict = norm[0]
            edge_weight_dict = norm[1]
            num_nodes_dict = norm[2]
            special_dict = norm[3]

        # compute and cache alpha_dict
        alpha_dict: Optional[NodeTypeFloatDict] = self._alpha_dict
        if alpha_dict is None:
            alpha_dict = gen_alpha_dict(list(x_dict.keys()), alpha_exp_dict)
            if self.cached:
                self._alpha_dict = alpha_dict

        # compute and cache beta_dict
        beta_dict: Optional[EdgeTypeFloatDict] = self._beta_dict
        if beta_dict is None:
            beta_dict = gen_beta_dict(
                list(edge_index_dict.keys()), beta_exp_dict
            )
            if self.cached:
                self._beta_dict = beta_dict

        # update x_dict if special_nodes were added
        for node_type, special in special_dict.items():  # type: ignore
            x = x_dict[node_type]
            size = list(x.size())
            if special == size[0]:
                size[0] = 1
                x_dict[node_type] = torch.cat((x, x.new_full(size, 0)), dim=0)

        h_dict = x_dict.copy()

        for _ in range(self.K):
            _x_dict: NodeTypeTensorDict = {}
            for edge_type, edge_index in edge_index_dict.items():
                source, _, target = edge_type

                _x_dict[target] = _x_dict.get(
                    target,
                    alpha_dict[target] * h_dict[target],
                )

                x = x_dict[source]
                edge_weight = edge_weight_dict[edge_type]  # type: ignore
                size = [num_nodes_dict[source], num_nodes_dict[target]]
                _x_dict[target] += (
                    (1 - alpha_dict[target])
                    * beta_dict[edge_type]
                    * self.propagate(
                        edge_index,
                        x=x,
                        edge_weight=edge_weight,
                        size=size,
                    )
                )
            x_dict = _x_dict.copy()

        return x_dict

    @staticmethod
    def message(x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(K={self.K})"
