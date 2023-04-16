
import os
import glob
from typing import List, Optional, Tuple, Union
import logging
import shutil

logger = logging.getLogger(__name__)

import bilby
import yaml

from .. import utils
from ..jeans import binned_jeans_model, binned_lp_model, density_profiles
from ..jeans import dist_functions

class JeansInferenceModel():
    """ Sample the dark matter density from kinematic data using Jeans equation.
    Fit the light profile first, then use the fitted parameters as the initial
    guess for the joint fit of the dark matter density profile and the light profile.

    The light profile is fitted using the Plummer model. The DM density profile
    is fitted using the generalized NFW model. The distribution function is
    assumed to be the Osipkov-Merritt distribution function.

    """

    def __init__(
            self,
            run_name: str,
            config_file: Optional[str] = None,
            run_prefix: Optional[str] = None,
            priors: Optional[dict] = None,
            model_params: Optional[dict] = None,
            resume: bool = False,
        ):
        """

        Parameters
        ----------
        run_name: str
            Name of the run
        config_file: str
            Path to the config file
        run_prefix: str
            Prefix of the run
        priors: dict
            Dictionary of priors. Overwrite config file `priors` if specified
        model_params: dict
            Dictionary of model parameters. Overwrite config file `model` if specified
        resume: bool
            Whether to resume the run
        """
        if run_prefix is None:
            run_prefix = ''

        if config_file is not None:
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.priors = config['priors']
            self.model_params = config['model']
            # overwrite priors and model params if specified
            self.priors.update(priors)
            self.model_params.update(model_params)
        else:
            self.model_params = model_params
            self.priors = priors

        self.run_name = run_name
        self.run_prefix = run_prefix
        self.output_dir = None
        self.priors = priors
        self.dm_density_profile = None
        self.lp_density_profile = None
        self.dist_function = None

        self._setup_dir(resume=resume)
        self._setup_model()
        self._setup_bilby_priors()

    def _setup_model(self):
        """ Set up the model parameters and priors """
        self.dm_density_profile = density_profiles.GeneralizedNFW
        self.lp_density_profile = density_profiles.Plummer
        self.dist_function = dist_functions.OsipkovMerritt
        self.parameters = {
            "dm": self.dm_density_profile.PARAMETERS,
            "lp": self.lp_density_profile.PARAMETERS,
            "df": self.dist_function.PARAMETERS,
            "other": {}  # for other free parameters
        }
        if self.model_params['fit_v_mean']:
            self.parameters['other']['v_mean'] = {}

    def _setup_dir(self, resume: bool = False):
        """ Set up the output directory and write all params into yaml """
        # create an output directory
        # overwrite existing directory if not resuming
        self.output_dir = os.path.join(self.run_prefix, self.run_name)
        if not resume:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

        # write all params into yaml
        params = {
            'run_name': self.run_name,
            'run_prefix': self.run_prefix,
            'lp_density_profile': self.lp_density_profile.__name__,
            'dm_density_profile': self.dm_density_profile.__name__,
            'dist_function': self.dist_function.__name__,
        }
        with open(
            os.path.join(self.output_dir, 'params.yaml'),
            'w', encoding='utf-8') as f:
            yaml.dump(params, f, default_flow_style=False)

    def _setup_bilby_priors(self):
        """ Parse the priors from a dictionary to a Bilby prior dictionary """
        bilby_priors = bilby.core.prior.PriorDict()
        for key, val in self.priors.items():
            if val['type'] == 'Uniform':
                bilby_priors[key] = bilby.core.prior.Uniform(
                    val['min'], val['max'], name=key)
            elif val['type'] == 'LogUniform':
                bilby_priors[key] = bilby.core.prior.LogUniform(
                    val['min'], val['max'], name=key)
            elif val['type'] == 'Normal':
                bilby_priors[key] = bilby.core.prior.Normal(
                    val['mean'], val['std'], name=key)
            elif val['type'] == 'LogNormal':
                bilby_priors[key] = bilby.core.prior.LogNormal(
                    val['mean'], val['std'], name=key)
            elif val['type'] == 'DeltaFunction':
                bilby_priors[key] = bilby.core.prior.DeltaFunction(
                    val['value'], name=key)
            else:
                raise ValueError(f"Unknown prior type: {val['type']}")
        self.bilby_priors = bilby_priors

    def _get_bilby_priors(self, key):
        """ Get the priors for a given key """
        return {k: self.bilby_priors[k] for k in self.parameters[key]}

    def _get_bilby_jeans_priors(self, lp_model):
        """ Get the joint priors for all Jeans parameters. The fitted light profile
        parameters are used as the initial guess for the joint fit of the dark matter
        density profile and the light profile.
        """
        # get a bilby prior with all the parameters
        bilby_priors = self.bilby_priors.copy()

        # replace the light profile parameters with the fitted value
        for k, v in lp_model.items():
            bilby_priors[k] = bilby.core.prior.DeltaFunction(v, name=k)

    def fit(self):
        raise NotImplementedError

    def sample(
            self, data,
            data_name: str = 'data',
            sampler: str = 'dynesty',
            sampler_args: Optional[dict] = None,
            return_lp_model: bool = False,
        ):
        """ Perform a joint fit of the dark matter density profile and the light profile.
        First, fit the light profile using the Plummer model. Then, use the fitted
        parameters as the initial guess for the joint fit of the dark matter density
        profile and the light profile.

        Parameters
        ----------
        data: dict of ndarray
            Dictionary of kinematic data. Must includes `pos`, `vel`, `vel_error`.
        data_name: str
            Name of the data. Output file will be saved to `self.output_dir/data_name`.
        sampler: str
            Sampler to use
        sampler_args: dict
            Arguments to pass to the sampler
        return_lp_model: bool
            Whether to return the light profile model
        """
        if sampler_args is None:
            sampler_args = {}
        data_outdir = os.path.join(self.output_dir, data_name)

        # First, we define and sample the light profile model
        lp_priors  = self._get_bilby_priors('lp')
        lp_model = binned_lp_model.BinnedLPModel(
            self.lp_density_profile, data, priors=lp_priors)
        lp_model.run_sampler(
            sampler=sampler, label="lp_fit", outdir=data_outdir,
            **sampler_args
        )

        # Then, we define and sample the dark matter density profile model
        dm_priors = self._get_dm_priors_from_lp(lp_model)
        dm_model = binned_jeans_model.BinnedJeansModel(
            dm_profile=self.dm_density_profile,
            lp_profile=self.lp_density_profile,
            dist_function=self.dist_function,
            data=data, priors=dm_priors
        )
        dm_model.run_sampler(
            sampler=sampler, label="dm_fit", outdir=data_outdir,
            **sampler_args
        )

        # return the light profile model if requested
        if return_lp_model:
            return lp_model, dm_model
        return dm_model