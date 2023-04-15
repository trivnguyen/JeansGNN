
from typing import Optional

import numpy as np
import scipy.integrate as integrate


class DensityProfile:
    """ Base class for density profile """

    PARAMETERS = ()

    def __init__(self, parameters: dict):
        """
        Parameters
        ----------
        parameters: dict
            The parameters of the density profile
        """
        self.parameters = parameters

    def log_density(self, r: np.ndarray) -> np.ndarray:
        """ Compute the log10 density profile """
        raise NotImplementedError

    def density(self, r: np.ndarray, *args, **kargs) -> np.ndarray:
        """ Compute the density profile """
        return 10**self.log_density(r, *args, **kargs)

    def cumulative_mass(self, r: np.ndarray) -> np.ndarray:
        """ Compute the enclosed mass at each radius """
        # check if the array r is equally spaced
        dr = r[1] - r[0]
        if np.any(np.abs(np.diff(r) - dr) > 1e-6):
            return integrate.cumtrapz(self.density(r), dx=dr, initial=0)
        return integrate.cumtrapz(self.density(r), r, initial=0)


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

    def log_density(self, r: np.ndarray) -> np.ndarray:
        """ Compute the log density of the generalized NFW profile.
        Equation:
        ```
        log10 rho(r) = log10 rho_0 - gamma * log10(r / r_dm) + (3 - gamma) * log10(1 + r / r_dm)
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
        return np.log10(rho_0) - (gamma) * np.log10(x) + (3 - gamma) * np.log10(1 + x)


class Plummer(DensityProfile):
    """ The Plummer profile. Can be used for 2D or 3D."""

    PARAMETERS = ("L", "r_star")

    def __init__(self, L, r_star, projected: bool = False):
        super().__init__(parameters={
            "L": L,
            "r_star": r_star,
        })
        self.projected = projected

    def log_density(
            self, r: np.ndarray, projected: Optional[bool]=None) -> np.ndarray:
        """ Compute the log density of the Plummer profile.

        Parameters
        ----------
        r: np.ndarray
            The radius
        projected: bool
            Whether the radius is projected or not. If True, the 2D Plummer profile
            will be used. If False, the 3D Plummer profile will be used.
            If None, the value of self.projected will be used.

        Returns
        -------
        np.ndarray
            The log density
        """
        if projected is None:
            projected = self.projected

        if projected:
            return self.log_density2d(r)
        return self.log_density3d(r)

    def log_density2d(self, r: np.ndarray) -> np.ndarray:
        """ Compute the log density of the 2D Plummer profile.
        Here, r is the projected radius instead of the 3D radius. Equation:
        ```
        log10 rho(r) = log10 L - 2 log10 r_star - 2 log10 (1 + r^2 / r_star^2) - log10 pi
        ```
        where:
            L: luminosity
            r_star: scale radius
        """
        L = self.parameters["L"]
        r_star = self.parameters["r_star"]
        logL = np.log10(L)
        logr_star = np.log10(r_star)
        x = r / r_star
        return logL - 2 * logr_star - 2 * np.log10(1 + x**2) - np.log10(np.pi)

    def log_density3d(self, r: np.ndarray) -> np.ndarray:
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
        logL = np.log10(L)
        logr_star = np.log10(r_star)
        x = r / r_star
        return logL - 3 * logr_star - (5/2) * np.log10(1 + x**2) - np.log10(4 * np.pi / 3)

