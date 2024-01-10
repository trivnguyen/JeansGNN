
from typing import List

import torch
from .base_transform import BaseTransform

class UniformNoise(BaseTransform):
    """ Add uniform error to the input features """
    def __init__(
        self, min_error: float, max_error: float,
        index: List[int] = None, concat_error: bool = False):
        """
        Parameters
        ----------
        min_error: float
            Minimum error to add to the input features
        max_error: float
            Maximum error to add to the input features
        index: List[int]
            List of indices to add the error to. If None, will add error to all
            dimensions
        concat_error: bool
            If True, will concatenate the error to the input features
        """
        super().__init__()
        if min_error > max_error:
            raise ValueError("min_error must be less than max_error")
        if min_error < 0:
            raise ValueError("min_error must be positive")
        if isinstance(index, int):
            index = [index]

        self.min_error = torch.tensor(min_error, dtype=torch.float32)
        self.max_error = torch.tensor(max_error, dtype=torch.float32)
        self.index = torch.tensor(index, dtype=torch.long)
        self.concat_error = concat_error

    def __call__(self, batch):
        """ Add errors to the input features of batch"""
        batch = batch.clone()
        with torch.no_grad():
            self.index = self.index.to(batch.x.device)
            x = torch.index_select(batch.x, 1, self.index)
            max_error = self.max_error.to(x.device)
            min_error = self.min_error.to(x.device)

            # generate random errors and convolve with the input features
            errors = torch.rand_like(x) * (max_error - min_error) + min_error
            x_errors = torch.randn_like(x) * errors + x

            # replace the input features with the convolved features
            batch.x[:, self.index] = x_errors

            # concatenate the errors to the input features
            if self.concat_error:
                batch.x = torch.cat([batch.x, errors], axis=-1)
        return batch

    def recompute_indim(self, indim: int) -> int:
        """ Recompute the input dimension after the transformation """
        if self.concat_error:
            return indim + len(self.index)
        else:
            return indim


class GaussianNoise(BaseTransform):
    """ Add Gaussian error to the input features """
    def __init__(
        self, mean: float, std: float,
        index: List[int] = None, concat_error: bool = False):
        """
        Parameters
        ----------
        mean: float
            Mean of the Gaussian error to add to the input features
        std: float
            Standard deviation of the Gaussian error to add to the input features
        index: List[int]
            List of indices to add the error to. If None, will add error to all
            dimensions
        concat_error: bool
            If True, will concatenate the error to the input features
        """
        super().__init__()
        if isinstance(index, int):
            index = [index]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.index = torch.tensor(index, dtype=torch.long)
        self.concat_error = concat_error

    def __call__(self, batch):
        """ Add errors to the input features of batch"""
        batch = batch.clone()
        with torch.no_grad():
            self.index = self.index.to(batch.x.device)
            x = torch.index_select(batch.x, 1, self.index)
            mean = self.mean.to(x.device)
            std = self.std.to(x.device)

            # generate random errors and convolve with the input features
            errors = torch.randn_like(x) * std + mean
            errors = torch.clamp(errors, min=0) # errors must be positive
            x_errors = errors + x

            # replace the input features with the convolved features
            batch.x[:, self.index] = x_errors

            # concatenate the errors to the input features
            if self.concat_error:
                batch.x = torch.cat([batch.x, errors], axis=-1)
        return batch

    def recompute_indim(self, indim: int) -> int:
        """ Recompute the input dimension after the transformation """
        if self.concat_error:
            return indim + len(self.index)
        else:
            return indim
