
from typing import Callable
import torch

def get_activation(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """ Parse the activation function to use in the MAF model """
    if activation == 'tanh':
        return torch.tanh
    elif activation == 'relu':
        return torch.relu
    elif activation == 'sigmoid':
        return torch.sigmoid
    else:
        raise RuntimeError(
            "activation should be tanh/relu/sigmoid, not {}".format(activation))
