
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data


class ExponentialEdgeWeightDistance(T.BaseTransform):
    """ Compute edge weight as exponential of distance between nodes

    Parameters
    ----------
    norm: bool
        Normalize the edge weight by the mean distance

    Attributes
    ----------
    norm: bool
        Normalize the edge weight by the mean distance

    Methods
    -------
    __call__(data)
        Compute edge weight
    """

    def __init__(self, norm: bool = False):
        self.norm = False

    def __call__(self, data: Data):
        x1 = data.pos[data.edge_index[0]]
        x2 = data.pos[data.edge_index[1]]
        d = torch.linalg.norm(x1-x2, ord=2, dim=1)
        if self.norm:
            rho = torch.mean(d)
            data.edge_weight = torch.exp(-d**2 / rho**2)
        else:
            data.edge_weight = torch.exp(-d**2)
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm})')

class PhaseSpaceGraphProcessor():
    """
    Process phase space data into graph data

    Attributes
    ----------
    GRAPH_DICT: dict
        Dictionary of graph types

    Methods
    -------
    """
    GRAPH_DICT = {
        "KNNGraph": T.KNNGraph,
        "RadiusGraph": T.RadiusGraph,
    }
    EDGE_WEIGHT_DICT = {
        "exp": ExponentialEdgeWeightDistance,
    }

    def __init__(
            self,
            graph_name: str = "KNNGraph",
            log_radius: bool = False,
            edge_weight_name: Optional[str] = None,
            edge_weight_params: Optional[dict] = None
        ):
        """
        Parameters
        ----------
        graph_name: str
            Name of the graph to use
        log_radius: bool
            Logarithmize the radius
        edge_weight_name: str
            Name of the edge weight function
        edge_weight_params: dict
            Parameters for the edge weight function
        """
        self.graph_name = graph_name
        self.log_radius = log_radius
        self.edge_weight_name = edge_weight_name
        self.edge_weight_params = edge_weight_params

        # parse graph name
        self.graph = self._parse_graph_name(graph_name)
        self.edge_weight = self._parse_edge_weight_name(edge_weight_name)

    def __call__(self, pos: Tensor, vel: Tensor,
                 label: Optional[Tensor] = None):
        """
        Parameters
        ----------
        pos: torch.Tensor
            Position tensor
        vel: torch.Tensor
            Velocity tensor
        label: torch.Tensor
            Label tensor. If None, return data without label
        """
        # transform into 1D feature vector
        x = self.feature_preprocess(pos, vel)

        # create graph
        data = Data(pos=pos, x=x)
        data = self.graph(pos=data.pos, **self.graph_params)
        data = self.edge_weight(data, **self.edge_weight_params)

        # add label
        if label is not None:
            data.y = label
        return data

    def feature_preprocess(self, pos: Tensor, vel: Tensor):
        """
        Transform 2D position and velocity into 1D feature vector

        Parameters
        ----------
        pos: torch.Tensor
            Position tensor
        vel: torch.Tensor
            Velocity tensor
        """
        radius = torch.linalg.norm(pos, ord=2, dim=1, keepdims=True)
        if self.log_radius:
            return torch.hstack([torch.log10(radius), vel]).squeeze()
        return torch.hstack([radius, vel]).squeeze()

    def _parse_graph_name(self, graph_name):
        if graph_name in self.GRAPH_DICT:
            return self.GRAPH_DICT[graph_name]
        else:
            raise KeyError(
                f"Unknown graph name \"{graph_name}\"."\
                f"Available models are: {str(self.GRAPH_DICT.keys())}")

    def _parse_edge_weight_name(self, edge_weight_name):
        if edge_weight_name is None:
            return lambda x: x

        if edge_weight_name in self.EDGE_WEIGHT_DICT:
            return self.EDGE_WEIGHT_DICT[edge_weight_name]
        else:
            raise KeyError(
                f"Unknown edge weight name \"{edge_weight_name}\"."\
                f"Available models are: {str(self.EDGE_WEIGHT_DICT.keys())}")
