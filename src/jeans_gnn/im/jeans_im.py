
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


def parse_bilby_priors(priors: dict):
    """ Parse the priors from a dictionary to a Bilby prior dictionary """
    parse_dict = {
        'Uniform': bilby.core.prior.Uniform,
        'LogUniform': bilby.core.prior.LogUniform,
        'Normal': bilby.core.prior.Normal,
        'LogNormal': bilby.core.prior.LogNormal,
        'DeltaFunction': bilby.core.prior.DeltaFunction,
    }
    bilby_priors = bilby.core.prior.PriorDict()
    for key, val in priors.items():
        prior_type = val.pop('type')
        if prior_type not in parse_dict:
            raise ValueError(f'Unknown prior type: {prior_type}')
        bilby_priors[key] = parse_dict[prior_type](name=key, **val)
    return bilby_priors

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
        if priors is None:
            priors = {}
        if model_params is None:
            model_params = {}

        if config_file is not None:
            with open(config_file, 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            self.priors = config['priors']
            self.model_params = config['model']
            # overwrite priors and model params if specified
            self.priors.update(priors)
            self.model_params.update(model_params)
        else:
            self.priors = priors
            self.model_params = model_params

        self.run_name = run_name
        self.run_prefix = run_prefix
        self.output_dir = None
        self.dm_density_profile = None
        self.lp_density_profile = None
        self.dist_function = None

        self._setup_model()
        self._setup_bilby_priors()
        self._setup_dir(resume=resume)

    def _setup_model(self):
        """ Set up the model parameters and priors """
        self.dm_density_profile = density_profiles.GeneralizedNFW
        self.lp_density_profile = density_profiles.Plummer
        self.dist_function = dist_functions.OsipkovMerritt
        self.parameters = {
            "dm": list(self.dm_density_profile.PARAMETERS),
            "lp": list(self.lp_density_profile.PARAMETERS),
            "df": list(self.dist_function.PARAMETERS),
            "other": [],  # for other free parameters
        }
        if self.model_params['jeans_fit']['fit_v_mean']:
            self.parameters['other'] += ['v_mean']

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
            'model': self.model_params,
            'priors': self.priors,
        }
        with open(
            os.path.join(self.output_dir, 'params.yaml'),
            'w', encoding='utf-8') as f:
            yaml.dump(params, f, default_flow_style=False)

    def _setup_bilby_priors(self):
        """ Parse the priors from a dictionary to a Bilby prior dictionary """
        # check if priors ia already a bilby prior dict
        if isinstance(self.priors, bilby.core.prior.PriorDict):
            self.bilby_priors = self.priors
        else:
            self.bilby_priors = parse_bilby_priors(self.priors)

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
        lp_prior_type = self.model_params.get('lp_prior_type', 'best_median')
        for key in self.parameters['lp']:
            if  lp_prior_type == 'ci':
                val_lo, val_hi = lp_model.get_credible_intervals(key, p=0.95)
                bilby_priors[key] = bilby.core.prior.Uniform(val_lo, val_hi, key)
            elif lp_prior_type == 'normal':
                mean, std = lp_model.get_mean_and_std(key)
                bilby_priors[key] = bilby.core.prior.Gaussian(mean, std, key)
            elif lp_prior_type == 'best_median':
                median = lp_model.get_median(key)
                bilby_priors[key] = bilby.core.prior.DeltaFunction(median, key)
            elif lp_prior_type == 'best_mean':
                mean = lp_model.get_mean(key)
                bilby_priors[key] = bilby.core.prior.DeltaFunction(mean, key)
            elif lp_prior_type == 'uni':
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown lp_prior_type in `model_params`: {lp_prior_type}")
        return bilby_priors

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
            self.lp_density_profile, data, priors=lp_priors,
            **self.model_params.get('lp_fit', {})
        )
        lp_model.run_sampler(
            sampler=sampler, label="lp_fit", outdir=data_outdir,
            **sampler_args
        )

        # Then, we define and sample the dark matter density profile model
        dm_priors = self._get_bilby_jeans_priors(lp_model)
        dm_model = binned_jeans_model.BinnedJeansModel(
            dm_profile=self.dm_density_profile,
            lp_profile=self.lp_density_profile,
            dist_function=self.dist_function,
            data=data, priors=dm_priors,
            **self.model_params.get('dm_fit', {})
        )
        dm_model.run_sampler(
            sampler=sampler, label="dm_fit", outdir=data_outdir,
            **sampler_args
        )

        # return the light profile model if requested
        if return_lp_model:
            return dm_model, lp_model
        return dm_model