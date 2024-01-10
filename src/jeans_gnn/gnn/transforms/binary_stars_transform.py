

from typing import List

import torch
import astropy.units as u

from .base_transform import BaseTransform

# Constants
AU_TO_KM = u.au.to(u.km)
DAY_TO_SEC = u.day.to(u.s)


def calc_binary_vlos(q, P, a, sin_i, phi):
    """ Calculate the pbserved line-of-sight velocity of a binary system.

    Parameters
    ----------
    q: float or array_like
        Mass ratio of the binary system.
    P: float or array_like
        Orbital period of the binary system in days.
    a: float or array_like
        Semi-major axis of the binary system in AU.
    sin_i: float or array_like
        Sine of the inclination angle of the binary system.
    phi: float or array_like
        Orbital phase of the binary system in radians.

    Returns
    -------
    vlos1: float or array_like
        Observed line-of-sight velocity of the primary star in km/s.
    vlos2: float or array_like
        Observed line-of-sight velocity of the secondary star in km/s.

    """
    # convert a to km and P to seconds
    a = a * AU_TO_KM
    P = P * DAY_TO_SEC

    # Calculate the LOS velocity of the primary and secondary stars
    vlos1 = (2 * torch.pi * a / P) * (1 / (1 + q)) * sin_i * torch.sin(phi)
    vlos2 = (2 * torch.pi * a / P) * (q / (1 + q)) * sin_i * torch.sin(phi)
    return vlos1, vlos2

class AddBinaryPopulation(BaseTransform):
    """ Add a binary population to the dataset """

    def __init__(
        self, binary_params: dict, index: List[int] = None,
        concat_bin_frac: bool = False, concat_v_binary: bool = False
    ):
        """ Initialize the transform

        Parameters
        ----------
        binary_params: dict
            Dictionary of binary parameters.
        index: List[int]
            List of indices to add the binary population to.
        concat_bin_frac: bool
            Whether to concatenate the binary fraction to the labels.
        concat_v_binary: bool
            Whether to concatenate the binary velocity to the velocities.
        """
        if isinstance(index, int):
            index = [index]

        self.binary_params = binary_params
        self.index = torch.tensor(index, dtype=torch.long)
        self.concat_bin_frac = concat_bin_frac
        self.concat_v_binary = concat_v_binary

    def __call__(self, batch):
        """ Add a binary population to the velocities """
        batch = batch.clone()

        with torch.no_grad():
            num_stars = batch.x.shape[0]
            num_galaxies = batch.batch[-1] + 1
            self.index = self.index.to(batch.x.device)

            # Sample the binary parameters
            a, q, P, sin_i, phi, bin_frac = self._sample_binary_params(
                num_stars, num_galaxies)

            # Calculate the binary velocity and randomly pick from the two velocities
            vbin1, vbin2 = calc_binary_vlos(q, P, a, sin_i, phi)
            bin_mask = torch.randint(0, 2, (num_stars,))
            vbin = bin_mask * vbin1 + (1 - bin_mask) * vbin2

            # randomly set to zeros based on the binary fraction
            # batch by batch
            for i in range(num_galaxies):
                # get the indices of the stars in the batch
                n = batch.ptr[i + 1] - batch.ptr[i]
                bin_mask = torch.rand(n) > bin_frac[i]
                vbin[batch.ptr[i]: batch.ptr[i + 1]][bin_mask] = 0
            vbin = vbin.unsqueeze(-1).to(batch.x.device)

            # add the binary velocity to the systemic velocity
            vsys = torch.index_select(batch.x, 1, self.index)
            batch.x[:, self.index] = vsys + vbin

            # concatenate the binary fraction to the labels
            if self.concat_bin_frac:
                bin_frac = bin_frac[:, None].to(batch.x.device)
                batch.y = torch.cat((batch.y, bin_frac), dim=1)

            # concatenate the binary velocity to the velocities
            if self.concat_v_binary:
                batch.x = torch.cat((batch.x, vbin), dim=1)
        return batch

    def _sample_binary_params(self, num_stars: int, num_galaxies: int):
        """ Sample the binary parameters """
        with torch.no_grad():
            # population parameter
            bin_frac = self._sample_from_dist(
                num_galaxies, self.binary_params['bin_frac'])

            # individual binary parameters
            # assume isotropic distribution and uniform phase
            a = self._sample_from_dist(num_stars, self.binary_params['a'])
            q = self._sample_from_dist(num_stars, self.binary_params['q'])
            P = self._sample_from_dist(num_stars, self.binary_params['P'])
            sin_i = torch.rand(num_stars)
            phi = torch.rand(num_stars) * 2 * torch.pi

            return a, q, P, sin_i, phi, bin_frac

    def _sample_from_dist(self, num_samples, dist_params):
        """ sample from a distribution """
        if dist_params['type'] == 'uniform':
            min_val = torch.tensor(dist_params['min'], dtype=torch.float32)
            max_val = torch.tensor(dist_params['max'], dtype=torch.float32)
            return torch.rand(num_samples) * (max_val - min_val) + min_val
        elif dist_params['type'] == 'log_uniform':
            min_val = torch.tensor(dist_params['min'], dtype=torch.float32)
            max_val = torch.tensor(dist_params['max'], dtype=torch.float32)
            log_min_val = torch.log(min_val)
            log_max_val = torch.log(max_val)
            return torch.exp(
                torch.rand(num_samples) * (log_max_val - log_min_val) + log_min_val)
        elif dist_params['type'] == 'normal':
            mean = torch.tensor(dist_params['mean'], dtype=torch.float32)
            std = torch.tensor(dist_params['std'], dtype=torch.float32)
            return torch.randn(num_samples) * std + mean
        elif dist_params['type'] == 'delta':
            value = torch.tensor(dist_params['value'], dtype=torch.float32)
            return torch.ones(num_samples) * value
        else:
            raise ValueError('Unknown distribution type')

    def recompute_indim(self, indim: int) -> int:
        if self.concat_v_binary:
            return indim + 1
        return indim

    def recompute_outdim(self, outdim: int) -> int:
        if self.concat_bin_frac:
            return outdim + 1
        return outdim