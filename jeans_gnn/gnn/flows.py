
from typing import Callable

import torch
import torch.nn.functional as F
from nflows import distributions, transforms, flows

def _get_activation(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Parse activation function """
    if activation == 'tanh':
        return torch.tanh
    elif activation == 'relu':
        return torch.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    else:
        raise RuntimeError(
            "activation should be tanh/relu/sigmoid, not {}".format(activation))

def build_maf(
        features: int, hidden_features: int, context_features: int,
        num_layers: int, num_blocks: int, activation: str = 'tanh'
    ) -> flows.Flow:
    """ Build masked autoregressive flow (MAF) layers

    Parameters
    ----------
    features : int
        Number of features
    hidden_features : int
        Number of hidden features
    context_features : int
        Number of context features
    num_layers : int
        Number of MAF layers
    num_blocks : int
        Number of MAF blocks
    activation : str
        Activation function

    Returns
    -------
    maf : nflows.flows.Flow
        Masked autoregressive flow models
    """
    transform = []
    transform.append(transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=features,
                        hidden_features=hidden_features,
                        context_features=context_features,
                        num_blocks=num_blocks,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation=_get_activation(activation),
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=features),
                ]
            )
            for _ in range(num_layers)
        ]
    ))
    transform = transforms.CompositeTransform(transform)
    distribution = distributions.StandardNormal((features,))
    maf = flows.Flow(transform, distribution)
    return maf
