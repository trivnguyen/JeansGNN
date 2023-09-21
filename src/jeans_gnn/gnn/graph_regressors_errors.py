
from typing import Callable, Optional, Union, List

import numpy as np
import torch
import torch_geometric
from torch_geometric.nn import ChebConv, GATConv, GCNConv

from .base_modules import BaseFlowModule
from .flows import build_maf
from .graph_regressors import GraphRegressor


class UniformErrorLayer(torch.nn.Module):
    """ Add error to the input features """
    def __init__(
            self, min_error: List, max_error: List,
            ignore_dims: Optional[List] = None):
        """
        Parameters
        ----------
        min_error: list
            Minimum error with length (input_dimension)
        max_error: torch.Tensor
            Maximum error with length (input_dimension)
        ignore_dims: list of int
            Dimensions to ignore
        """

        super().__init__()
        self.register_buffer('min_error', torch.tensor(min_error, dtype=torch.float32))
        self.register_buffer('max_error', torch.tensor(max_error, dtype=torch.float32))

        self.ignore_dims = ignore_dims
        self.select_dims = np.arange(self.min_error.shape[0])
        # get all the chosen dimensions
        if self.ignore_dims is not None:
            self.select_dims = np.delete(self.select_dims, self.ignore_dims)

    def forward(self, x, return_error=True):
        """ Add errors to the input features of any shape """
        max_error = self.max_error.to(x.device)
        min_error = self.min_error.to(x.device)
        errors = torch.rand_like(x) * (max_error - min_error) + min_error
        if self.ignore_dims is not None:
            errors[..., self.ignore_dims] = 0

        # convolve the errors with the input features
        x = torch.randn_like(x) * errors + x
        if return_error:
            if self.ignore_dims is not None:
                return x, errors[..., self.select_dims]
            return x, errors
        else:
            return x


class GaussianErrorLayer(torch.nn.Module):
    """ Add error to the input features """
    def __init__(
            self, mean: List, stdv: List,
            ignore_dims: Optional[List] = None):
        """
        Parameters
        ----------
        mean: List
            Mean error with length (input_dimension)
        stdv: List
            Standard deviation with length (input_dimension)
        ignore_dims: list of ints
            Dimensions to ignore
        """
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('stdv', torch.tensor(stdv, dtype=torch.float32))

        self.ignore_dims = ignore_dims
        self.select_dims = np.arange(self.mean.shape[0])
        # get all the chosen dimensions
        if self.ignore_dims is not None:
            self.select_dims = np.delete(self.select_dims, self.ignore_dims)


    def forward(self, x, return_error=True):
        """ Add errors to the input features of any shape """
        # using the reparameterization trick
        mean = self.mean.to(x.device)
        stdv = self.stdv.to(x.device)
        errors = torch.randn_like(x) * stdv + mean
        errors = torch.clamp(errors, min=0)

        if self.ignore_dims is not None:
            errors[..., self.ignore_dims] = 0

        # convolve the errors with the input features
        x = torch.randn_like(x) * errors + x
        if return_error:
            if self.ignore_dims is not None:
                return x, errors[..., self.select_dims]
            return x, errors
        else:
            return x


class GraphRegressorWithErrors(GraphRegressor):
    """ Graph regressors with feature errors """

    ERROR_LAYERS = {
        'Uniform': UniformErrorLayer,
        'Gaussian': GaussianErrorLayer,
    }

    def __init__(
            self, in_channels: int, error_layer_name: str,
            error_layer_params: Optional[dict] = None, *args, **kargs):
        """
        Parameters
        ----------
        in_channels: int
            Number of input channels before concatenating with the error
        error_layer: str
            Name of the error layer
        error_layer_dict: dict
            Dictionary of arguments for the error layer. Must contain the following keys:
            - ignore_dims: list of ints
                Dimensions to ignore
        *args: positional arguments
            Positional arguments for the parent class
        **kargs: keyword
            Keyword arguments for the parent class
        """
        # add error dimensions to the input channels
        if 'ignore_dims' not in error_layer_params:
            error_layer_params['ignore_dims'] = None
            in_channels += in_channels
        else:
            in_channels += in_channels - len(error_layer_params['ignore_dims'])
        super().__init__(in_channels, *args, **kargs)

        # get the error layer
        self.error_layer = self._get_error_layer(error_layer_name, error_layer_params)

    def _add_feature_errors(self, x: torch.Tensor) -> torch.Tensor:
        """ Add feature errors to the input features.
        Will also concatenate the errors to the input features """
        x, error = self.error_layer(x)
        x = torch.cat([x, error], axis=-1)
        return x

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
        # Add error to the input features before applying the graph layers
        x = self._add_feature_errors(x)
        x = self.forward_with_error(x, edge_index, batch, edge_weight)
        return x

    def forward_with_error(self, x, edge_index, batch, edge_weight=None):
        """ Forward but x already includes the error """
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

    def _get_error_layer(self, error_layer_name, error_layer_params):
        """ Return the error layer

        Parameters
        ----------
        in_dim: int
            Input dimension
        out_dim: int
            Output dimension
        error_layer_name: str
            Name of the error layer
        error_layer_params: dict
            Parameters of the error kayer

        Returns
        -------
        error_layer: torch.nn.Module
        """
        if error_layer_name not in self.ERROR_LAYERS:
            raise ValueError(f"Error layer {error_layer_name} not implemented")
        return self.ERROR_LAYERS[error_layer_name](
            **error_layer_params)

class GraphRegressorWithErrorsModule(BaseFlowModule):
    """ Graph Regressor module """
    def __init__(
            self, model_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None,
            scheduler_hparams: Optional[dict] = None,
        ) -> None:
        super().__init__(
            GraphRegressorWithErrors, model_hparams, optimizer_hparams, scheduler_hparams)
