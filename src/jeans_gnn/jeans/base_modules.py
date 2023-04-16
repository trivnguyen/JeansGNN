
import bilby
import numpy as np


class BilbyModule(bilby.Likelihood):
    """ Wrapper for bilby.Likelihood with a few useful functions

    Attributes
    ----------
    result : bilby.core.result.Result
        The result of the sampler
    priors : dict
        The priors of the parameters

    Methods
    -------
    log_likelihood()
        Log likelihood
    run_sampler(*args, **kargs)
        Use bilby to run the sampler
    get_credible_intervals(key, p=0.95)
        Return the credible intervals of the posterior of a given key
    get_median(key)
        Return the median of the posterior of a given key
    get_mean_and_std(key)
        Return the mean and standard deviation of the posterior of a given key
    """
    def __init__(self, parameters):
        super().__init__(parameters=parameters)
        """

        Parameters
        ----------
        parameters : dict
            bilby.Likelihood requires a dict of parameters
        """
        self.result = None
        self.priors = {}

    def log_likelihood(self):
        raise NotImplementedError

    def run_sampler(self, *args, **kargs):
        self.result = bilby.run_sampler(
            likelihood=self, priors=self.priors, *args, **kargs)

    def get_credible_intervals(self, key, p=0.95):
        """ Return the credible intervals of the posterior of a given key """
        lo = (1 - p) / 2 * 100
        hi = (1 + p) / 2 * 100
        values = self.result.posterior[key].values
        return np.percentile(values, q=[lo, hi])

    def get_median(self, key):
        """ Return the median of the posterior of a given key """
        values = self.result.posterior[key].values
        return np.percentile(values, q=50)

    def get_mean_and_std(self, key):
        """ Return the mean and standard deviation of the posterior of a given key """
        values = self.result.posterior[key].values
        return np.mean(values), np.std(values)

    def load_result(self, *args, **kargs):
        self.result = bilby.core.result.read_in_result(*args, **kargs)