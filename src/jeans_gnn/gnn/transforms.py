
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_geometric.transforms as T
from torch import Tensor
from torch_geometric.data import Data


class ExponentialEdgeWeightDistance(T.BaseTransform):
    """ Compute edge weight as exponential of distance between nodes

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
        """
        Parameters
        ----------
        norm: bool
            Normalize the edge weight by the mean distance
        """
        super().__init__()
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
            graph_params: Optional[dict] = None,
            radius: bool = True,
            log_radius: bool = True,
            edge_weight_name: Optional[str] = None,
            edge_weight_params: Optional[dict] = None,
            tensor_type: Union[str, torch.dtype] = torch.float32,
        ):
        """
        Parameters
        ----------
        graph_name: str
            Name of the graph to use
        graph_params: dict
            Parameters for the graph
        radius: bool
            Use radius as features
        log_radius: bool
            Logarithmize the radius
        edge_weight_name: str
            Name of the edge weight function
        edge_weight_params: dict
            Parameters for the edge weight function
        """
        if graph_params is None:
            graph_params = {}
        if edge_weight_params is None:
            edge_weight_params = {}
        self.graph_name = graph_name
        self.graph_params = graph_params
        self.radius = radius
        self.log_radius = log_radius
        self.edge_weight_name = edge_weight_name
        self.edge_weight_params = edge_weight_params
        self.tensor_type = tensor_type

        # parse graph name
        self.graph = self._parse_graph_name(graph_name, **graph_params)
        self.edge_weight = self._parse_edge_weight_name(
            edge_weight_name, **edge_weight_params)

        # store dimensions
        self.graph_params = {
            'in_channels': 2,
            'out_channels': 1,
        }

    def __call__(self, pos: Tensor, vel: Tensor,
                 label: Optional[Tensor] = None):
        """
        Parameters
        ----------
        pos: torch.Tensor or numpy.ndarray
            Position tensor. Shape (N, 2)
        vel: torch.Tensor or numpy.ndarray
            Velocity tensor. Shape (N, 1)
        label: torch.Tensor
            Label tensor. If None, return data without label
        """
        # convert from numpy array to torch
        if isinstance(pos, np.ndarray):
            pos = torch.tensor(pos, dtype=self.tensor_type)
        if isinstance(vel, np.ndarray):
            vel = torch.tensor(vel, dtype=self.tensor_type)
        if label is not None and isinstance(label, np.ndarray):
            label = torch.tensor(label, dtype=self.tensor_type)
            if label.ndim == 1:
                label = label.reshape(1, -1)

        # transform into 1D feature vector
        x = self.feature_preprocess(pos, vel)

        # create graph
        data = Data(pos=pos, x=x)
        data = self.graph(data)
        data = self.edge_weight(data)

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
        # reshape vel to (N, 1) if 1 dimensional
        if vel.ndim == 1:
            vel = vel.reshape(-1, 1)

        if self.radius or self.log_radius:
            radius = torch.linalg.norm(pos, ord=2, dim=1, keepdims=True)
            if self.log_radius:
                return torch.hstack([torch.log10(radius), vel])
            return torch.hstack([radius, vel])
        else:
            return torch.hstack([pos, vel])

    def _parse_graph_name(self, graph_name, **kwargs):
        if graph_name in self.GRAPH_DICT:
            return self.GRAPH_DICT[graph_name](**kwargs)
        else:
            raise KeyError(
                f"Unknown graph name \"{graph_name}\"."\
                f"Available models are: {str(self.GRAPH_DICT.keys())}")

    def _parse_edge_weight_name(self, edge_weight_name, **kwargs):
        if edge_weight_name is None:
            return lambda x: x

        if edge_weight_name in self.EDGE_WEIGHT_DICT:
            return self.EDGE_WEIGHT_DICT[edge_weight_name](**kwargs)
        else:
            raise KeyError(
                f"Unknown edge weight name \"{edge_weight_name}\"."\
                f"Available models are: {str(self.EDGE_WEIGHT_DICT.keys())}")
