# APPrOVE

[![Tests](https://github.com/mrmrob003/approve/actions/workflows/tests.yml/badge.svg)](https://github.com/mrmrob003/approve/actions/workflows/test.yml) [![codecov](https://codecov.io/gh/mrmrob003/approve/graph/badge.svg?token=79PPMLYSBT)](https://codecov.io/gh/mrmrob003/approve) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 😎 Summary

A [PyG](https://pytorch-geometric.readthedocs.io/en/latest/index.html) (PyTorch Geometric) implementation of the propagation model for heterogeneous graphs detailed in ["APPROVE: Approximate Personalized Propagation over Varied Edges"](https://arxiv.org/abs/23xx.xxxxx). 

## 🧠 Theory

For more information about personalized PageRank and its extension to heterogeneous graphs, read ["APPROVE: Approximate Personalized Propagation Over Varied Edges"](https://arxiv.org/abs/23xx.xxxxx).

## 🚀 Installation

The most recent release can be installed from
[PyPI](https://pypi.org/project/approve/) with:

```bash
$ pip install approve
```

The most recent code can be installed directly from GitHub with:

```bash
$ pip install git+https://github.com/mrmrob003/approve.git
```

## 🏃 Getting Started
To demonstrate our heterogeneous personalized PageRank algorithm, consider the following toy-model of a citation network consisting of three papers and two venues.

<p align="center">
  <img src="examples/citation_network.png">
</p>

```python
hetero_data = HeteroData()
hetero_data['paper', 'cites', 'paper'].edge_index = torch.tensor(
    [[1, 2, 2],
     [0, 0, 1]]
)
hetero_data['venue', 'publishes', 'paper'].edge_index = torch.tensor(
    [[0, 1],
     [0, 1]]
)
hetero_data['paper', 'rev_publishes', 'venue'].edge_index = \
    hetero_data['venue', 'publishes', 'paper'].edge_index[[1,0]]
hetero_data['paper'].num_nodes = 3
hetero_data['venue'].num_nodes = 2
```

Paper `0` is cited by the other two papers and published by venue `0`, while paper `1` is cited by paper `2` and published by venue `1`.

To compute the type-level PageRank score of each node, we initially assign uniform scores to all nodes of a given type. Since there are three papers and two venues, we assign each paper a third of the total `'paper'` score and each venue half of the total `'venue'` score.

```python
hetero_data['paper'].x = torch.full((3, 1), 1 / 3)
hetero_data['venue'].x = torch.full((2, 1), 1 / 2)
```

Furthermore, we need to add self-loops to `'paper'` nodes, and a special edge from paper `2` (which is unpublished) to a special `'venue'` node. The addition of the self-loops and special edge prevents the scores for each node type from leaking.

<p align="center">
  <img src="examples/citation_network_updated.png">
</p>

The `approve.models.HeteroAPPR` model takes care of all these considerations and can be used to compute the score of each node as follows.

```python
model = HeteroAPPR(K=30)
output = model(
    hetero_data.x_dict, 
    edge_index_dict=hetero_data.edge_index_dict,
)
output
```
```
{'paper': tensor([[0.4605],
         [0.3289],
         [0.2105]]),
 'venue': tensor([[0.4803],
         [0.4145],
         [0.1053]])}
```

Unsurprisingly, paper `0` is the most important paper, since it is cited by the other two papers. Venues `0` and `1` have comparable scores; venue `0`'s score is slighlty larger than venue `1`'s score, because venue `0` publishes a higher-ranked paper than the paper published by venue `1`. Venue `2`, the special `'venue'` node, has a comparably low score because it relates to the lowest ranked paper.

## 👋 Attribution

### ⚖️ License
The code in this package is licensed under the [MIT License](./LICENSE).

### 📖 Citation
If you use this software in your work, please cite it using the "Cite this repository" widget located on the sidebar.
