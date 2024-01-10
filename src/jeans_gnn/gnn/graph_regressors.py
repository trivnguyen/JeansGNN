
from typing import Callable, Optional, Union

import torch
import torch_geometric
from torch_geometric.nn import ChebConv, GATConv, GCNConv

from .base_modules import BaseFlowModule
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
    graph_layers: torch.nn.ModuleList
        List of graph layers
    fc_layers: torch.nn.ModuleList
        List of fully connected layers
    activation: torch.nn.Module or Callable or str
        Activation function
    activation_params: dict
        Parameters of the activation function.
    flows: torch.nn.ModuleList
        List of normalizing flow layers

    Methods
    -------
    forward(x, edge_index, batch, edge_weight=None)
        Forward pass of the model
    log_prob(batch, return_context=False)
        Calculate log-likelihood from batch
    sample(batch, num_samples, return_context=False)
        Sample from batch
    log_prob_from_context(x, context)
        Calculate log-likelihood P(x | context)
    sample_from_context(num_samples, context)
        Sample from context

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
            hidden_fc_channels: int = 1,
            num_fc_layers: int = 1,
            graph_layer_name: str = "ChebConv",
            graph_layer_params: Optional[dict] = None,
            layer_norm: bool = False,
            activation: Union[str, torch.nn.Module, Callable] = "relu",
            activation_params: Optional[dict] = None,
            flow_params: dict = None
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
        hidden_fc_channels: int
            Hidden dimension of the fully connected layers
        num_fc_layers: int
            Number of fully connected layers
        graph_layer_name: str
            Name of the graph layer
        graph_layer_params: dict
            Parameters of the graph layer
        layer_norm: bool
            Whether to use layer normalization
        activation: str or torch.nn.Module or Callable
            Activation function
        activation_params: dict
            Parameters of the activation function. Ignored if activation is
            torch.nn.Module
        flow_params: dict
            Parameters of the normalizing flow
        """
        super().__init__()

        if graph_layer_params is None:
            graph_layer_params = {}
        if activation_params is None:
            activation_params = {}

        self.graph_layer_name = graph_layer_name

        # Create the graph layers
        self.graph_layers = torch.nn.ModuleList()
        for i in range(num_graph_layers):
            n_in = in_channels if i == 0 else hidden_graph_channels
            n_out = hidden_graph_channels
            self.graph_layers.append(
                self._get_graph_layer(
                    n_in, n_out, graph_layer_name, graph_layer_params))

        # add layer normalization
        if layer_norm:
            self.graph_layers_norm = torch.nn.ModuleList()
            for i in range(num_graph_layers):
                self.graph_layers_norm.append(
                    torch_geometric.nn.LayerNorm(hidden_graph_channels))
        else:
            self.graph_layers_norm = None

        # Create FC layers
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            n_in = hidden_graph_channels if i == 0 else hidden_fc_channels
            n_out = hidden_fc_channels
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        # Create activation function
        if isinstance(activation, str):
            self.activation = getattr(torch.nn.functional, activation)
            self.activation_params = activation_params
        elif isinstance(activation, torch.nn.Module):
            self.activation = activation
            self.activation_params = {}
        elif isinstance(activation, Callable):
            self.activation = activation
            self.activation_params = activation_params
        else:
            raise ValueError("Invalid activation function")

        # Create MAF normalizing flow layers
        self.flows = build_maf(
            channels=out_channels, context_channels=hidden_fc_channels,
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
        for i, layer in enumerate(self.graph_layers):
            if self.HAS_EDGE_WEIGHT[self.graph_layer_name]:
                x = layer(x, edge_index, edge_weight=edge_weight)
            else:
                x = layer(x, edge_index)
            if self.graph_layers_norm is not None:
                x = self.graph_layers_norm[i](x)
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

    def log_prob(self, batch, return_context=False, forward_args=None):
        """ Calculate log-likelihood from batch """
        if forward_args is None:
            forward_args = {}
        context = self.forward(
            batch.x, batch.edge_index, batch.batch,
            edge_weight=batch.edge_weight, **forward_args)
        log_prob = self.flows.log_prob(batch.y, context=context)

        if return_context:
            return log_prob, context
        return log_prob

    def sample(
        self, batch, num_samples, return_context=False, forward_args=None):
        """ Sample from batch """
        if forward_args is None:
            forward_args = {}
        with torch.no_grad():
            context = self.forward(
                batch.x, batch.edge_index, batch.batch,
                edge_weight=batch.edge_weight, **forward_args)

            y = self.flows.sample(num_samples, context=context)

            if return_context:
                return y, context
            return y

    def log_prob_from_context(self, x, context):
        """ Return MAF log-likelihood P(x | context)"""
        return self.flows.log_prob(x, context=context)

    def sample_from_context(self, num_samples, context):
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

class GraphRegressorModule(BaseFlowModule):
    """ Graph Regressor module """
    def __init__(
            self, model_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None,
            scheduler_hparams: Optional[dict] = None,
            pre_transform_hparams: Optional[dict] = None,
        ) -> None:
        super().__init__(
            GraphRegressor, model_hparams, optimizer_hparams, scheduler_hparams,
            pre_transform_hparams)
