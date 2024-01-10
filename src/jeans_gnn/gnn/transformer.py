
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .base_modules import BaseFlowModule
from .flows import build_maf

class TransformerRegressor(nn.Module):
    """
    Featurizer based on the TransformerEncoder module from PyTorch.

    Attributes
    ----------
    embedding : nn.Linear
        The embedding layer.
    transformer_encoder : nn.TransformerEncoder
        The transformer encoder.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            dim_feedforward: int,
            use_embedding: bool = True,
            hidden_fc_channels: int = 128,
            num_fc_layers: int = 2,
            activation: Union[str, torch.nn.Module, Callable] = "relu",
            activation_params: Optional[dict] = None,
            flow_params: dict = None
        ):
        """
        Parameters
        ----------
        in_channels: int
            Input dimension
        out_channels: int
            Output dimension
        d_model: int
            Dimension of the embedding
        nhead: int
            Number of heads in the multihead attention models
        num_encoder_layers: int
            Number of sub-encoder-layers in the encoder
        dim_feedforward: int
            Dimension of the feedforward network model
        use_embedding: bool
            Whether to use an embedding layer
        hidden_fc_channels: int
            Hidden dimension of the fully connected layers
        num_fc_layers: int
            Number of fully connected layers
        activation: str or torch.nn.Module or Callable
            Activation function
        activation_params: dict
            Parameters of the activation function. Ignored if activation is
            torch.nn.Module
        flow_params: dict
            Parameters of the normalizing flow
        """
        super().__init__()

        if use_embedding:
            self.embedding = nn.Linear(in_channels, d_model)
        else:
            assert in_channels == d_model, (
                "If not using embedding, input_size must be equal to d_model."
                f"Got input_size={input_size} and d_model={d_model}"
            )
            self.embedding = nn.Identity()

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers)

        # Create FC layers
        self.fc_layers = torch.nn.ModuleList()
        for i in range(num_fc_layers):
            n_in = dim_feedforward if i == 0 else hidden_fc_channels
            n_out = hidden_fc_channels
            self.fc_layers.append(torch.nn.Linear(n_in, n_out))

        # Create activation function
        if isinstance(activation, str):
            self.activation = getattr(torch.nn.functional, activation)
            self.activation_params = activation_params or {}
        elif isinstance(activation, torch.nn.Module):
            self.activation = activation
            self.activation_params = {}
        elif isinstance(activation, Callable):
            self.activation = activation
            self.activation_params = activation_params or {}
        else:
            raise ValueError("Invalid activation function")

        # Create MAF normalizing flow layers
        self.flows = build_maf(
            channels=out_channels, context_channels=hidden_fc_channels,
            **flow_params)


    def forward(self, x, padding_mask=None):
        x = self.embedding(x)
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)

        # NOTE: dimension only works when batch_first=True
        if padding_mask is None:
            output = output.sum(dim=1)
        else:
            if not self.training:
                # apply correct padding mask for evaluation
                # this happens because with both self.eval() and torch.no_grad()
                # the transformer encoder changes the length of the output to
                # match the max non-padded length in the batch
                max_seq_len = torch.max(padding_mask.eq(0).sum(-1))
                padding_mask = padding_mask[:, :max_seq_len]
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
            output = output.sum(dim=1)

        # apply FC layers
        # do not apply activation function to the last layer
        for layer in self.fc_layers[:-1]:
            x = layer(x)
            x = self.activation(x, **self.activation_params)
        x = self.fc_layers[-1](x)

        return output

    def log_prob(self, batch, return_context=False, forward_args=None):
        """ Calculate log-likelihood from batch """
        if forward_args is None:
            forward_args = {}
        x = batch[0]
        y = batch[1]
        padding_mask = batch[2]

        try:
            context = self.forward(x, padding_mask, **forward_args)
            log_prob = self.flows.log_prob(y, context=context)
        except:
            print(x.shape, y.shape, padding_mask.shape)
            raise

        if return_context:
            return log_prob, context
        return log_prob

    @torch.no_grad()
    def sample(
        self, batch, num_samples, return_context=False, forward_args=None):
        """ Sample from batch """
        if forward_args is None:
            forward_args = {}
        x = batch[0]
        y = batch[1]
        padding_mask = batch[2]

        context = self.forward(x, padding_mask, **forward_args)
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


class TransformerRegressorModule(BaseFlowModule):
    """ Transformer Regressor module """
    def __init__(
            self, model_hparams: Optional[dict] = None,
            optimizer_hparams: Optional[dict] = None,
            scheduler_hparams: Optional[dict] = None,
            pre_transform_hparams: Optional[dict] = None,
        ) -> None:
        super().__init__(
            TransformerRegressor, model_hparams, optimizer_hparams, scheduler_hparams,
            pre_transform_hparams)
