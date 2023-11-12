

import logging
from typing import Optional

logger = logging.getLogger(__name__)

import jax.numpy as jnp
import astropy.constants as const
import astropy.units as u

from . import utils
from .density_profiles import DensityProfile
from .dist_functions import DistributionFunction


class BinnedJeansModel():
    """
    Class for fitting the DM density profile using binned Jeans modeling
    Implementation is based on Chang & Necib (arXiv:2009.00613)

    Attributes
    ----------
    REQUIRED_DATA_KEYS : tuple
        The required keys in the `data` dictionary
    dm_profile: DensityProfile
        The DM density profile class
    lp_profile: DensityProfile
        The light profile density class
    dist_function: DistributionFunction
        The distribution function class
    data: dict
        Dictionary containing the kinematics data
    priors: dict
        The priors in bilby format
    r_min_factor: float
        Factor to convert the min projected radius R to the min 3D radius
    r_max_factor: float
        Factor to convert the max projected radius R to the max 3D radius
    dr: float
        The radius integration resolution
    """

    REQUIRED_DATA_KEYS = ("pos", "vel", "vel_error")

    def __init__(
        self,
        dm_profile: DensityProfile,
        lp_profile: DensityProfile,
        dist_function: DistributionFunction,
        data: dict,
        r_min_factor: float = 0.5,
        r_max_factor: float = 2,
        dr: float = 0.001,
        fit_v_mean: bool = True
    ):
        """
        Parameters
        ----------
        dm_profile: DensityProfile
            The DM density profile class
        lp_profile: DensityProfile
            The light profile density class
        dist_function: DistributionFunction
            The distribution function class
        data: dict
            Dictionary containing the kinematics data
        priors: dict
            The priors in bilby format
        r_min_factor: float
            Factor to convert the min projected radius R to the min 3D radius
        r_max_factor: float
            Factor to convert the max projected radius R to the max 3D radius
        dr: float
            The radius integration resolution
        fit_v_mean: bool
            Whether to fit for the mean velocity. If not, the mean velocity is
            calculated from the projected velocity in `data['vel']`
        """
        # define parameters
        parameters = []
        parameters += dm_profile.PARAMETERS
        parameters += lp_profile.PARAMETERS
        parameters += dist_function.PARAMETERS
        parameters += ["v_mean"]

        # if there is any duplication in the parameters list, raise an error
        if len(parameters) != len(set(parameters)):
            raise ValueError(
                "There are duplicated parameters in the parameters list")

        # check if the data has the required keys
        for key in self.REQUIRED_DATA_KEYS:
            if key not in data:
                raise KeyError(f"Data dictionary does not contain {key}")

        # initialize the base class
        super().__init__()
        

        # define attributes
        self.dm_profile = dm_profile
        self.lp_profile = lp_profile
        self.dist_function = dist_function
        self.data = data
        self.r_min_factor = r_min_factor
        self.r_max_factor = r_max_factor
        self.dr = dr
        self.fit_v_mean = fit_v_mean
        self._unit_norm = const.G.to_value(u.kpc**3 / u.Msun / u.s**2) 
        self._unit_norm *= (u.kpc).to(u.km)**2
        self.parameters = {k: None for k in parameters}

        # set up likelihood function
        self._setup_likelihood()

    def _setup_likelihood(self):
        """ Setup before running the likelihood function """
        pos = self.data['pos']
        vel = self.data['vel']
        vel_error = self.data['vel_error']
        radius = jnp.linalg.norm(pos, axis=1)

        # define integration radius array
        r_min = jnp.min(radius) * self.r_min_factor
        r_max = jnp.max(radius) * self.r_max_factor
        r_arr = jnp.arange(r_min, r_max + self.dr, self.dr)

        # calculate the mean velocity if not fitting for it
        # set priors to Delta function

        # add prepared data as attributes
        self.vel_var = vel_error**2
        self.int_radius = r_arr
        self.data['radius'] = radius

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
        """ The log likelihood given a set of DM parameters.
        For each star the log likelihood is defined as:
        .. math::
        logL = -0.5 * (v - v_mean)^2 / (sigma2_p + v_err^2) - 0.5 * log(2 pi  * (sigma2_p + verr^2))

        where:
        - v is the velocity of the star
        - v_mean is the mean velocity of all stars
        - v_err is the measurement error
        - sigma2_p is the velocity dispersion
        """
        # get parameters and construct profiles
        dm_params = {k: self.parameters[k] for k in self.dm_profile.PARAMETERS}
        lp_params = {k: self.parameters[k] for k in self.lp_profile.PARAMETERS}
        dist_params = {k: self.parameters[k] for k in self.dist_function.PARAMETERS}
        dm_profile = self.dm_profile(**dm_params)
        lp_profile = self.lp_profile(**lp_params)
        dist_function = self.dist_function(**dist_params)
        v_mean = self.parameters['v_mean']

        # First, we calculate the projected velocity dispersion profile
        # calculate the velocity ani Beta(r) and the anisotropy integral g(r)
        beta = dist_function.velocity_anisotropy(self.int_radius)
        gint = utils.calc_gint(self.int_radius, beta)

        # calculate the light profile at each particle radius
        Sigma = jnp.float_power(
            10., lp_profile.log_density2d(self.data['radius']))

        # calculate the DM density profile at each integration radius
        nu = jnp.float_power(
            10., lp_profile.log_density3d(self.int_radius))

        # integrate the 3d Jeans velocity dispersion equation
        sigma2_nu = utils.calc_sigma2_nu(
            self.int_radius, dm_profile.cumulative_mass(self.int_radius), nu, gint)

        # integrate the 2d velocity dispersion equation
        sigma2p_Sigma = utils.calc_sigma2p_Sigma(
            self.data['radius'], self.int_radius, sigma2_nu, beta)
        sigma2p = sigma2p_Sigma / Sigma *self._unit_norm

        # calculate the log likelihood from the velocity dispersion
        # and the velocity measurement error
        var = sigma2p + self.vel_var
        logL = -0.5 * (self.data['vel'] - v_mean)**2 / var
        logL = logL - 0.5 * jnp.log(2 * jnp.pi * var)
        logL = jnp.sum(logL)

        return logL