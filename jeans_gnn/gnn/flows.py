

from nflows import distributions, transforms, flows

from .utils import get_activation

def build_maf(
        features: int, hidden_features: int, context_features: int,
        num_layers: int, num_blocks: int, activation: str = 'tanh'
    ) -> flows.Flow:
    """ Build a Masked Autoregressive Flow (MAF) model

    Parameters
    ----------
    features : int
        The number of features in the input data
    hidden_features : int
        The number of hidden features in the MAF model
    context_features : int
        The number of context features in the MAF model
    num_layers : int
        The number of layers in the MAF model
    num_blocks : int
        The number of blocks in each layer of the MAF model (i.e. the number of
        autoregressive transformations in each layer)
    activation : str
        The activation function to use in the MAF model. Should be one of
        'tanh', 'relu', or 'sigmoid'

    Returns
    -------
    maf : nflows.Flow
        The MAF model
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
                        activation= get_activation(activation),
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
