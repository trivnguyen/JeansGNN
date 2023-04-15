
import numpy as np

class DistributionFunction:
    """ Base class for distribution function """

    PARAMETERS = ()

    def __init__(self, parameters: dict):
        """
        Parameters
        ----------
        parameters: dict
            The parameters of the density profile
        """
        self.parameters = parameters

    def velocity_anisotropy(self, r: np.ndarray) -> np.ndarray:
        """ Compute the velocity anisotropy """
        raise NotImplementedError

class OsipkovMerrit(DistributionFunction):
    """ The Osipkov-Merrit distribution function """

    PARAMETERS = ("beta_0", "r_a")

    def __init__(self, beta_0: float = 1.0, r_a: float = 1.0):
        super().__init__(parameters={
            "beta_0": beta_0,
            "r_a": r_a,
        })

    def velocity_anisotropy(self, r: np.ndarray) -> np.ndarray:
        """ Compute the velocity anisotropy of the Osipkov-Merrit distribution function.
        Equation:
        ```
        beta(r) = beta_0 * (r / r_a)^2 / (1 + r / r_a)^2
        ```
        where:
            beta_0: central velocity anisotropy
            r_a: scale radius
        """
        beta_0 = self.parameters["beta_0"]
        r_a = self.parameters["r_a"]
        x = r / r_a
        return beta_0 * x**2 / (1 + x)**2