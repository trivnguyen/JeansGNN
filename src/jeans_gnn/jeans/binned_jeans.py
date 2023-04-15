

import logging
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np

from . import utils
from .base_modules import BilbyModule
from .density_profiles import DensityProfile
from .dist_functions import DistributionFunction

class BinnedLPModel(BilbyModule):
    """ Fit the light profile model

    Attributes
    ----------
    profile : DensityProfile
        The light profile density class
    radius : np.ndarray
        The projected radii of each star
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
    def __init__(
            self, profile: DensityProfile, radius: np.ndarray, priors: dict):
        """
        Parameters
        ----------
        profile : DensityProfile
            The light profile density class
        radius : np.ndarray
            The projected radii of each star
        priors : dict
            The priors in bilby format
        """
        super().__init__(
            parameters={k: None for k in profile.PARAMETERS})

        # define attributes
        self.radius = radius
        self.profile = profile
        self.priors = priors
        self.Sigma = None
        self.V1 = None
        self.V2 = None
        self.Rbins_ce = None

        # set up likelihood function
        self._setup_likelihood()

    def _setup_likelihood(self):
        """ Setup before running the likelihood function by calculating
        the light profile and the variance of the light profile
        """
        Sigma, Sigma_lo, Sigma_hi, logRbins_lo, logRbins_hi = utils.calc_Sigma(
            self.radius, alpha=0.32, return_bounds=True)
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


class BinnedJeansModel(BilbyModule):
    """
    Class for fitting the DM density profile using binned Jeans modeling
    Implementation is based on Chang & Necib (arXiv:2009.00613)
    """

    def __init__(
        self, dm_profile: DensityProfile, light_profile: DensityProfile,
        dist_function: DistributionFunction, radius2d: list, vel: list,
        priors: dict, vel_err: Optional[list] = None,
        r_min_factor: float = 0.5, r_max_factor: float = 2,
        dr: float = 0.001
    ):
        """
        Parameters
        ----------
        dm_profile: DensityProfile
            The DM density profile class
        light_profile: DensityProfile
            The light profile density class
        dist_function: DistributionFunction
            The distribution function class
        radius2d: array of N float
            The projected radii of N stars in kpc
        vel: array of N float
            The line-of-sight velocities of N stars in km/s
        priors: dict
            The priors in bilby format
        vel_err: array of N float
            The line-of-sight velocity errors of N stars in km/s
        r_min_factor: float
            Factor to convert the min projected radius R to the min 3D radius
        r_max_factor: float
            Factor to convert the max projected radius R to the max 3D radius
        dr: float
            The radius integration resolution
        """
        parameters_list = (
            dm_profile.PARAMETERS
            + light_profile.PARAMETERS
            + dist_function.PARAMETERS
            + ['v_mean']
        )
        super().__init__(parameters={
            k: None for k in parameters_list
        })

        if vel_err is None:
            vel_err = np.zeros_like(vel)

        # define attributes
        self.dm_profile = dm_profile
        self.light_profile = light_profile
        self.dist_function = dist_function
        self.radius2d = radius2d
        self.vel = vel
        self.priors = priors
        self.vel_err = vel_err
        self.r_min_factor = r_min_factor
        self.r_max_factor = r_max_factor
        self.dr = dr

    def _setup_likelihood(self):
        """ Setup before running the likelihood function """
        vel_var = self.vel_err**2
        r_min = np.min(self.radius) * self.r_min_factor
        r_max = np.max(self.radius) * self.r_max_factor
        r_arr = np.arange(r_min, r_max + self.dr, self.dr)

        self.radius3d = r_arr
        self.vel_var = vel_var

    def log_likelihood(self):
        """ The log likelihood given a set of DM parameters.
        For each star the log likelihood is defined as:
        ```
        logL = -0.5 * (v - v_mean)^2 / (sigma2_p + v_err^2) - 0.5 * log(2 pi  * (sigma2_p + verr^2))
        ``
        where:
        - v is the velocity of the star
        - v_mean is the mean velocity of all stars
        - v_err is the measurement error
        - sigma2_p is the velocity dispersion
        """
        # get parameters and construct profiles
        dm_parameters = {
            k: self.parameters[k] for k in self.dm_profile.PARAMETERS}
        light_parameters = {
            k: self.parameters[k] for k in self.light_profile.PARAMETERS}
        dist_function_parameters = {
            k: self.parameters[k] for k in self.dist_function.PARAMETERS}
        dm_profile = self.dm_profile(**dm_parameters)
        light_profile = self.light_profile(**light_parameters)
        dist_function = self.dist_function(**dist_function_parameters)
        v_mean = self.parameters['v_mean']

        # calculate the 3D and 2D light profile
        Sigma = light_profile.density(self.radius2d, projected=True)
        nu = light_profile.density(self.radius2d, projected=False)

        # calculate Beta(r) and g(r), the anisotropy integral
        beta = dist_function.velocity_anisotropy(self.radius3d)
        g = utils.calc_g(self.radius3d, beta)

        # calculate the projected 2d velocity dispersion
        # calculate the DM density and cumulative mass
        dm_cmass = dm_profile.cumulative_mass(self.radius3d)
        sigma2_nu = utils.calc_sigma2_nu(self.radius3d, dm_cmass, nu, g)
        sigma2p_Sigma = utils.calc_sigma2p_Sigma(
            self.radius2d, self.radius3d, sigma2_nu, beta)
        sigma2p = sigma2p_Sigma / Sigma * utils.kpc_to_km**2

        # calculate the log likelihood
        vel_rms = (self.vel - v_mean)**2
        var = sigma2p + self.vel_var
        logL = -0.5 * vel_rms / var
        logL = logL - 0.5 * np.log(2 * np.pi * var)
        logL = np.sum(logL)

        return logL
