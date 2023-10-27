from typing import Dict, List, Optional, TypeAlias

from torch import Tensor

from torch_geometric.typing import EdgeType, NodeType

OptInt: TypeAlias = Optional[int]

EdgeTypeFloatDict: TypeAlias = Dict[EdgeType, float]

EdgeTypeFloatOptDict: TypeAlias = Optional[EdgeTypeFloatDict]

EdgeTypeIntDict: TypeAlias = Dict[EdgeType, int]

EdgeTypeList: TypeAlias = List[EdgeType]

EdgeTypeTensorDict: TypeAlias = Dict[EdgeType, Tensor]

EdgeTypeTensorOptDict: TypeAlias = Optional[EdgeTypeTensorDict]

NodeTypeFloatDict: TypeAlias = Dict[NodeType, float]

NodeTypeFloatOptDict: TypeAlias = Optional[NodeTypeFloatDict]

NodeTypeIntDict: TypeAlias = Dict[NodeType, int]

NodeTypeIntOptDict: TypeAlias = Optional[NodeTypeIntDict]

NodeTypeOptIntDict: TypeAlias = Dict[NodeType, OptInt]

NodeTypeOptIntOptDict: TypeAlias = Optional[NodeTypeOptIntDict]

NodeTypeList: TypeAlias = List[NodeType]

NodeTypeTensorDict: TypeAlias = Dict[NodeType, Tensor]

NodeTypeTensorOptDict: TypeAlias = Optional[NodeTypeTensorDict]
