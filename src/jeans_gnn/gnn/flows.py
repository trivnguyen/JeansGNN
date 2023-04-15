
from nflows import distributions, transforms, flows

from .utils import get_activation

def build_maf(
        channels: int, hidden_channels: int, context_channels: int,
        num_layers: int, num_blocks: int, activation: str = 'tanh'
    ) -> flows.Flow:
    """ Build a MAF normalizing flow

    Parameters
    ----------
    channels: int
        Number of channels
    hidden_channels: int
        Number of hidden channels
    context_channels: int
        Number of context channels
    num_layers: int
        Number of layers
    num_blocks: int
        Number of blocks
    activation: str
        Name of the activation function
    """
    transform = []
    transform.append(transforms.CompositeTransform(
        [
            transforms.CompositeTransform(
                [
                    transforms.MaskedAffineAutoregressiveTransform(
                        features=channels,
                        hidden_features=hidden_channels,
                        context_features=context_channels,
                        num_blocks=num_blocks,
                        use_residual_blocks=False,
                        random_mask=False,
                        activation= get_activation(activation),
                        dropout_probability=0.0,
                        use_batch_norm=True,
                    ),
                    transforms.RandomPermutation(features=channels),
                ]
            )
            for _ in range(num_layers)
        ]
    ))
    transform = transforms.CompositeTransform(transform)
    distribution = distributions.StandardNormal((channels,))
    maf = flows.Flow(transform, distribution)
    return maf
