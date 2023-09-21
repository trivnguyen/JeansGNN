

import logging

logger = logging.getLogger(__name__)

import jax.numpy as jnp

from . import utils
from .density_profiles import DensityProfile


class BinnedLPModel():
    """ Fit the light profile model from kinematics data

    Attributes
    ----------
    REQUIRED_DATA_KEYS : tuple
        The required keys in the `data` dictionary
    profile : DensityProfile
        The light profile density class
    data: dict
        Dictionary containing the kinematics data
    Sigma : jnp.ndarray
        The surface density profile
    V1 : jnp.ndarray
        The first Poisson fit variance
    V2 : jnp.ndarray
        The second Poisson fit variance
    Rbins_ce : jnp.ndarray
        The center of the bins in log space
    """

    REQUIRED_DATA_KEYS = ("pos",)

    def __init__(
            self, profile: DensityProfile, data: dict):
        """
        Parameters
        ----------
        profile : DensityProfile
            The light profile density class
        data: dict
            Dictionary containing the kinematics data
        priors : dict
            The priors in bilby format
        """
        # check if the data has the required keys
        for key in self.REQUIRED_DATA_KEYS:
            if key not in data:
                raise KeyError(f"Data dictionary does not contain {key}")

        # initialize the base class
        super().__init__()

        # define attributes
        self.profile = profile
        self.data = data
        self.parameters = {k: None for k in profile.PARAMETERS}

        # set up likelihood function
        self._setup_likelihood()

    def _setup_likelihood(self):
        """ Setup before running the likelihood function by calculating
        the light profile and the variance of the light profile
        """
        pos = self.data['pos']
        radius_proj = jnp.linalg.norm(pos, axis=1)
        Sigma, Sigma_lo, Sigma_hi, logR_bins_lo, logR_bins_hi = utils.calc_Sigma(
            radius_proj, alpha=0.5)
        R_bins_hi = jnp.float_power(10., logR_bins_hi)
        R_bins_lo = jnp.float_power(10., logR_bins_lo)
        sig_lo = Sigma - Sigma_lo
        sig_hi = Sigma_hi - Sigma
        V1 = sig_lo * sig_hi
        V2 = sig_hi - sig_lo

        # store attributes
        self.Sigma = Sigma
        self.V1 = V1
        self.V2 = V2
        self.Rbins_ce = 0.5 * (R_bins_hi + R_bins_lo)

    def set_parameters(self, **kwargs):
        """ Set the parameters of the light profile

        Parameters
        ----------
        kwargs : dict
            The parameters of the light profile
        """
        for key, value in kwargs.items():
            if key not in list(self.parameters.keys()):
                raise KeyError(f"Parameter {key} is not defined")
            self.parameters[key] = value

    def log_likelihood(self):
        """ Log likelihood function defined as:
        ```
            logL = -0.5 * (Sigma - Sigma_hat)^2 / (V1 - V2 * (Sigma - Sigma_hat))
        ```
        where:
        - Sigma is the light profile as inferred from data
        - Sigma_hat is the estimated light profile
        - V1 and V2
        """
        profile = self.profile(**self.parameters)
        Sigma_hat = jnp.float_power(10., profile.log_density2d(self.Rbins_ce)) 
        delta_Sigma = self.Sigma - Sigma_hat
        return - 0.5 * jnp.sum(delta_Sigma**2 / (self.V1 - self.V2 * delta_Sigma))
