

import logging

logger = logging.getLogger(__name__)

import numpy as np

from . import utils
from .base_modules import BilbyModule
from .density_profiles import DensityProfile


class BinnedLPModel(BilbyModule):
    """ Fit the light profile model from kinematics data

    Attributes
    ----------
    REQUIRED_DATA_KEYS : tuple
        The required keys in the `data` dictionary
    profile : DensityProfile
        The light profile density class
    data: dict
        Dictionary containing the kinematics data
    priors : dict
        The priors in bilby format
    Sigma : np.ndarray
        The surface density profile
    V1 : np.ndarray
        The first Poisson fit variance
    V2 : np.ndarray
        The second Poisson fit variance
    Rbins_ce : np.ndarray
        The center of the bins in log space
    """

    REQUIRED_DATA_KEYS = ("pos",)

    def __init__(
            self, profile: DensityProfile, data: dict, priors: dict):
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
        super().__init__(
            parameters={k: None for k in profile.PARAMETERS})

        # define attributes
        self.profile = profile
        self.data = data
        self.priors = priors

        # set up likelihood function
        self._setup_likelihood()

    def _setup_likelihood(self):
        """ Setup before running the likelihood function by calculating
        the light profile and the variance of the light profile
        """
        pos = self.data['pos']
        radius = np.linalg.norm(pos, axis=1)

        Sigma, Sigma_lo, Sigma_hi, logRbins_lo, logRbins_hi = utils.calc_Sigma(
            radius, alpha=0.32, return_bounds=True)
        sig_lo = Sigma - Sigma_lo
        sig_hi = Sigma_hi - Sigma
        V1 = sig_lo * sig_hi
        V2 = sig_hi - sig_lo

        # store attributes
        self.Sigma = Sigma
        self.V1 = V1
        self.V2 = V2
        self.Rbins_ce = 10**(0.5 * (logRbins_lo + logRbins_hi))

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
        Sigma_hat = profile.density(self.Rbins_ce, projected=True)  # lp always projected
        delta_Sigma = self.Sigma - Sigma_hat
        return - 0.5 * np.sum(delta_Sigma**2 / (self.V1 - self.V2 * delta_Sigma))
