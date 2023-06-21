
from typing import Optional

import jax.numpy as jnp
from . import utils


class DensityProfile:
    """ Base class for density profile with Jax """
    PARAMETERS = ()

    def __init__(self, parameters: dict):
        """
        Parameters
        ----------
        parameters: dict
            The parameters of the density profile
        """
        self.parameters = parameters

    def __call__(self, r: jnp.array, *args, **kargs) -> jnp.array:
        """ Compute the density profile """
        return self.density(r, *args, **kargs)

    def density(self, r: jnp.array, *args, **kargs) -> jnp.array:
        """ Compute the density profile """
        return jnp.float_power(10., self.log_density(r, *args, **kargs))

    def cumulative_mass(self, r: jnp.array) -> jnp.array:
        """ Compute the enclosed mass at each radius """
        # check if the array r is equally spaced
        cmass = utils.jax_cumtrapz(
            4 * jnp.pi * r**2 * self.density(r), r)
        cmass = jnp.insert(cmass, -1, cmass[-1])
        return cmass


        return utils.jax_cumtrapz_init(self.density(r), r, initial=0)

    def log_density(self, r: jnp.array) -> jnp.array:
        """ Compute the log10 density profile """
        raise NotImplementedError


class GeneralizedNFW(DensityProfile):
    """ Generalized NFW profile """

    PARAMETERS = ("r_dm", "gamma", "rho_0")

    def __init__(self, r_dm: float = 1.0, gamma: float = 1.0,
                 rho_0: float = 1e6):
        super().__init__(parameters={
            "r_dm": r_dm,
            "gamma": gamma,
            "rho_0": rho_0,
        })

    def log_density(self, r: jnp.array) -> jnp.array:
        """ Compute the log density of the generalized NFW profile.
        Equation:
        ```
        log10 rho(r) = log10 rho_0 - gamma * log10(r / r_dm) - (3 - gamma) * log10(1 + r / r_dm)
        ```
        where:
            rho_0: central density
            r_dm: dark matter halo scale radius
            gamma: power law index
        """
        rho_0 = self.parameters["rho_0"]
        r_dm = self.parameters["r_dm"]
        gamma = self.parameters["gamma"]
        x = r / r_dm
        return jnp.log10(rho_0) - gamma * jnp.log10(x) - (3 - gamma) * jnp.log10(1 + x)


class Plummer(DensityProfile):
    """ The Plummer profile. Can be used for 2D or 3D."""

    PARAMETERS = ("L", "r_star")

    def __init__(self, L, r_star):
        super().__init__(parameters={
            "L": L,
            "r_star": r_star,
        })

    def log_density2d(self, r: jnp.array) -> jnp.array:
        """ Compute the log density of the 2D Plummer profile.
        Here, r is the projected radius instead of the 3D radius. Equation:
        ```
        log10 rho(r) = log10 L - 2 log10 r_star - 2 log10 (1 + r^2 / r_star^2) - log10 pi
        ```
        where:
            L: luminosity
            r_star: scale radius
        """
        # get parameters
        L = self.parameters["L"]
        r_star = self.parameters["r_star"]
        x = r / r_star
        return (
            jnp.log10(L) - 2 * jnp.log10(r_star) - 2 * jnp.log10(1 + x**2) 
            - jnp.log10(jnp.pi)
            )

    def log_density3d(self, r: jnp.array) -> jnp.array:
        """ Compute the log density of the 3D Plummer profile.
        Equation:
        ```
        log10 rho(r) = log10 L - 3 log10 r_star - 5/2 log10 (1 + r^2 / r_star^2) - log10 4 pi / 3
        ```
        where:
            L: luminosity
            r_star: scale radius
        """
        L = self.parameters["L"]
        r_star = self.parameters["r_star"]
        x = r / r_star
        return (
            jnp.log10(L) - 3 * jnp.log10(r_star) - (5/2) * jnp.log10(1 + x**2)
            - jnp.log10(4 * jnp.pi / 3))
