
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch_geometric
from torch_geometric.nn import ChebConv, GATConv, GCNConv

from .flows import build_maf


class DeepSet(torch.nn.Module):
    """ Simple DeepSet layer """
    def __init__(self, *args, **kargs):
        super().__init__()
        self.layer = torch.nn.Linear(*args, **kargs)

    def forward(self, x, edge_index, edge_weight=None):
        return self.layer(x)


class GraphRegressor(torch.nn.Module):
    """ Graph Regressor model

    Attributes
    ----------
    """
    # Static attributes
    # all implemented graph layers
    GRAPH_LAYERS = {
        "ChebConv": ChebConv,
        "GATConv": GATConv,
        "GCNConv": GCNConv,
        "DeepSet": DeepSet,
    }
    HAS_EDGE_WEIGHT = {
        "ChebConv": True,
        "GATConv": False,
        "GCNConv": True,
        "DeepSet": False,
    }

    def __init__(
            self, in_channels: int, out_channels: int,
            hidden_graph_channels: int = 1,
            num_graph_layers: int = 1,
            hidden_fc_layers: int = 1,
            num_fc_layers: int = 1,
            graph_layer_name: str = "ChebConv",
            graph_layer_params: dict = {},
            activation: str = "relu",
            activation_params: dict = {},
            flow_params: dict = {}
            ):
        """
        Parameters
        ----------
        in_channels: int
            Input dimension of the graph layers
        out_channels: int
            Output dimension of the normalizing flow
        hidden_graph_channels: int
            Hidden dimension
        num_graph_layers: int
            Number of graph layers
        hidden_fc_layers: int
            Hidden dimension of the fully connected layers
        num_fc_layers: int
            Number of fully connected layers
        graph_layer_name: str
            Name of the graph layer
        graph_layer_params: dict
            Parameters of the graph layer
        activation: str
            Name of the activation function
        activation_params: dict
            Parameters of the activation function
        flow_params: dict
            Parameters of the normalizing flow
        """

        super().__init__()

        self.graph_layer_name = graph_layer_name

        # Create the graph layers
        self.graph_layers = torch.nn.ModuleList()
        for i in range(num_graph_layers):
            n_in = in_channels if i == 0 else hidden_graph_channels
            n_out = hidden_graph_channels
            self.graph_layers.append(
                self._get_graph_layer(
                    n_in, n_out, graph_layer_name, graph_layer_params))

        # Create FC layers
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            n_in = hidden_fc_layers if i == 0 else hidden_fc_layers
            n_out = hidden_fc_layers
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        # Create activation function
        self.activation = getattr(torch.nn.functional, activation)
        self.activation_params = activation_params

        # Create MAF normalizing flow layers
        self.flows = build_maf(
            features=out_channels, context_features=hidden_fc_layers,
            **flow_params)

    def forward(self, x, edge_index, batch, edge_weight=None):
        """ Forward pass of the model

        Parameters
        ----------
        x: torch.Tensor
            Input features
        edge_index: torch.Tensor
            Edge indices
        edge_weight: torch.Tensor
            Edge weights

        Returns
        -------
        x: torch.Tensor
            Output features as the flows context
        """
        # Apply graph and FC layers to extract features as the flows context
        # apply graph layers
        for layer in self.graph_layers:
            if self.HAS_EDGE_WEIGHT[self.graph_layer_name]:
                x = layer(x, edge_index, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index)
            x = self.activation(x, **self.activation_params)
        # pool the features
        x = torch_geometric.nn.global_mean_pool(x, batch)

        # apply FC layers
        # do not apply activation function to the last layer
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.activation(x, **self.activation_params)
        x = self.fc_layers[-1](x)

        return x


    def log_prob(self, batch,return_context=False):
        """ Calculate log-likelihood from batch """
        context = self.forward(
            batch.x, batch.edge_index, batch.batch, edge_weight=batch.edge_weight)

        log_prob = self.flows.log_prob(batch.y, context=context)

        if return_context:
            return log_prob, context
        return log_prob

    def sample(self, batch, num_samples, return_context=False):
        """ Sample from batch """
        context = self.forward(
            batch.x, batch.edge_index, batch.batch, edge_weight=batch.edge_weight)

        y = self.flows.sample(num_samples, context=context)

        if return_context:
            return y, context
        return y

    def _log_prob_from_context(self, x, context):
        """ Return MAF log-likelihood P(x | context)"""
        return self.flows.log_prob(x, context=context)

    def _sample_from_context(self, num_samples, context):
        """ Sample P(x | context) """
        return self.flows.sample(num_samples, context=context)

    def _get_graph_layer(
            self, in_dim, out_dim, graph_layer_name, graph_layer_params):
        """ Return a graph layer

        Parameters
        ----------
        in_dim: int
            Input dimension
        out_dim: int
            Output dimension
        graph_layer_name: str
            Name of the graph layer
        graph_layer_params: dict
            Parameters of the graph layer

        Returns
        -------
        graph_layer: torch.nn.Module
        """
        if graph_layer_name not in self.GRAPH_LAYERS:
            raise ValueError(f"Graph layer {graph_layer_name} not implemented")
        return self.GRAPH_LAYERS[graph_layer_name](
            in_dim, out_dim, **graph_layer_params)
