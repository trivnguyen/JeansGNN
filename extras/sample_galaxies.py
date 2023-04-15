
# import modules
import argparse
import datetime
import logging
import os
import time

import agama
import astropy.units as u
import numpy as np
import pandas as pd
import yaml

import jeans_gnn as jgnn
from utils import envs, paths

# set agama unit to be in Msun, kpc, km/s
agama.setUnits(mass=1 * u.Msun, length=1*u.kpc, velocity=1 * u.km /u.s)

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, required=True,
        help='Path to the config file')
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to the output file. If not specified, the output will be '
             'saved to the default galaxy directory with the same name. The default '
             'directory is specified in the config file')
    parser.add_argument(
        '--name', type=str, default='default',
        help='Name of galaxy sets to generate')
    parser.add_argument(
        '--num-galaxies', type=int, default=10000,
        help='Number of galaxies to sample')
    return parser.parse_args()

# sample parameters from a config file
def sample_parameters(config, num_galaxies=10000):
    """ Sample parameters from a list of parameter configurations
    Parameters
    ----------
    config : list
        List of parameter configurations
    num_galaxies : int
        Number of galaxies to sample
    Returns
    -------
    parameters : dict
        Dictionary of sampled parameters
    """
    parameters = {}
    for c in config:
        name = c['name']
        dist = c['dist']
        if dist == 'uniform':
            parameters[name] = np.random.uniform(
                c['min'], c['max'], num_galaxies)
        elif dist == 'log_uniform':
            parameters[name] = 10**np.random.uniform(
                np.log10(c['min']), np.log10(c['max']), num_galaxies)
        elif dist == 'delta':
            parameters[name] = np.full(num_galaxies, c['value'])
        else:
            raise ValueError('Invalid distribution {}'.format(dist))
    # convert to DataFrame
    parameters = pd.DataFrame(parameters)
    return parameters

# parse sampled parameters into Agama format
def parse_parameters(
        dm_type, stellar_type, df_type,
        dm_params, stellar_params, df_params,
        return_params=False):
    """ Parse sampled parameters into Agama format
    Parameters
    ----------
    dm_type : str
        DM potential type
    stellar_type : str
        Stellar density profile type
    df_type : str
        DF type
    dm_params : dict
        DM potential parameters
    stellar_params : dict
        Stellar density profile parameters
    df_params : dict
        DF parameters
    return_params : bool
        Whether to return the parsed parameters
    Returns
    -------
    galaxy_model : agama.GalaxyModel
        Galaxy model
    params : dict
        Parsed parameters
    """
    # parse the DM parameters
    dm_params['densityNorm'] = dm_params.pop('rho_0')
    dm_params['scaleRadius'] = dm_params.pop('r_dm')
    dm_params['axisRatioY'] = dm_params.pop('q', 1)
    dm_params['axisRatioZ'] = dm_params.pop('p', 1)

    # parse the stellar parameters
    if stellar_params.get('r_star') is not None:
        stellar_params['scaleRadius'] = stellar_params.pop('r_star')
    elif stellar_params.get('r_star_r_dm') is not None:
        stellar_params['scaleRadius'] = (
            stellar_params.pop('r_star_r_dm') * dm_params['scaleRadius'])
    if stellar_params.get('q') is not None:
        stellar_params['axisRatioY'] = stellar_params.pop('q')
    elif stellar_params.get('q_star_q_dm') is not None:
        stellar_params['axisRatioY'] = (
            stellar_params.pop('q_star_q_dm') * dm_params['axisRatioY'])
    if stellar_params.get('p') is not None:
        stellar_params['axisRatioZ'] = stellar_params.pop('p')
    elif stellar_params.get('p_star_p_dm') is not None:
        stellar_params['axisRatioZ'] = (
            stellar_params.pop('p_star_p_dm') * dm_params['axisRatioZ'])

    # parse the DF parameters
    if df_params.get('r_a') is not None:
        df_params['r_a'] = df_params.pop('r_a')
    elif df_params.get('r_a_r_dm') is not None:
        df_params['r_a'] = (
            df_params.pop('r_a_r_dm') * dm_params['scaleRadius'])
    elif df_params.get('r_a_r_star') is not None:
        df_params['r_a'] = (
            df_params.pop('r_a_r_star') * stellar_params['scaleRadius'])

    # construct the DM potential, stellar density, and DF
    dm_potential = agama.Potential(
        type=dm_type, **dm_params)
    stellar_density = agama.Potential(
        type=stellar_type, mass=1, **stellar_params)  # set mass to small value
    df = agama.DistributionFunction(
        type=df_type, potential=dm_potential, density=stellar_density,
        **df_params)

    # construct galaxy model
    galaxy_model = agama.GalaxyModel(dm_potential, df)

    if return_params:
        # summarize params
        params = {'dm': dm_params, 'stellar': stellar_params, 'df': df_params}
        return galaxy_model , params
    return galaxy_model

def main():
    """ Sample the 6D stellar kinematics of dwarf galaxies """
    FLAGS = parse_args()

    # Load config file
    logger.info('Loading config file {}'.format(FLAGS.config))
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Sample parameters and number of stars
    logger.info('Sampling parameters for {} galaxies'.format(FLAGS.num_galaxies))
    dm_parameters = sample_parameters(
        config['dm_potential']['parameters'], FLAGS.num_galaxies)
    stellar_parameters = sample_parameters(
        config['stellar_density']['parameters'], FLAGS.num_galaxies)
    df_parameters = sample_parameters(
        config['distribution_function']['parameters'], FLAGS.num_galaxies)

    # sample number of stars
    if config['galaxy']['num_stars']['dist'] == 'uniform':
        num_stars = np.random.randint(
            config['galaxy']['num_stars']['min'],
            config['galaxy']['num_stars']['max'], FLAGS.num_galaxies)
    elif config['galaxy']['num_stars']['dist'] == 'poisson':
        num_stars = np.random.poisson(
            lam=config['galaxy']['num_stars']['mean'], size=FLAGS.num_galaxies)
    elif config['galaxy']['num_stars']['dist'] == 'delta':
        num_stars = np.full(
            FLAGS.num_galaxies, config['galaxy']['num_stars']['value'])
    else:
        raise ValueError('Invalid distribution for stars {}'.format(
            config['galaxy']['num_stars']['dist']))

    # Iterate over galaxies and sample the stellar kinematics
    logger.info('Sampling stellar kinematics')

    posvels = []
    for i in range(FLAGS.num_galaxies):
        # print progress every 10%
        if i % (FLAGS.num_galaxies // 10) == 0:
            logger.info('Galaxy {} of {}'.format(i, FLAGS.num_galaxies))

        # construct galaxy model
        gal = parse_parameters(
            config['dm_potential']['type'], config['stellar_density']['type'],
            config['distribution_function']['type'],
            dm_parameters.iloc[i].to_dict(),
            stellar_parameters.iloc[i].to_dict(),
            df_parameters.iloc[i].to_dict())

        # sample the stellar kinematics
        posvel_gal, _ = gal.sample(num_stars[i])

        # add parsed parameters and kinematics to list
        posvels.append(posvel_gal)

        # NOTE: there may be a memory leak in Agama, so we need to explicitly
        # delete the galaxy model
        del gal

    # convert kinematics to Numpy array
    posvels = np.array(posvels, dtype='object')

    # Prepare data for HDF5 file
    # create node features
    node_features = {}
    node_features['pos'] = np.concatenate(
        [posvels[i][:, :3] for i in range(len(posvels))])
    node_features['vel'] = np.concatenate(
        [posvels[i][:, 3:] for i in range(len(posvels))])

    # create graph features from parameters
    # Adding prefix to DataFrame column avoid name collisions
    dm_parameters = dm_parameters.add_prefix('dm_')
    stellar_parameters = stellar_parameters.add_prefix('stellar_')
    df_parameters = df_parameters.add_prefix('df_')
    parameters = pd.concat(
        [dm_parameters, stellar_parameters, df_parameters], axis=1)

    graph_features = parameters.to_dict('list')
    for k in graph_features:
        graph_features[k] = np.array(graph_features[k])
    graph_features['num_stars'] = num_stars

    # create headers
    headers = {
        'name': FLAGS.name,
        'node_features': list(node_features.keys()),
        'graph_features': list(graph_features.keys()),
        'dm_type': config['dm_potential']['type'],
        'stellar_type': config['stellar_density']['type'],
        'df_type': config['distribution_function']['type'],
        'num_galaxies': FLAGS.num_galaxies,
        'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Write kinematics to HDF5 file
    if FLAGS.output is None:
        # if no output file is specified, use the name from the config file
        # get config file name without extension
        config_name = os.path.splitext(os.path.basename(FLAGS.config))[0]

        # create output file name as "{name}_{config_name}.hdf5"
        FLAGS.output = os.path.join(
            envs.DEFAULT_GALAXIES_DIR,
            '{}_{}.hdf5'.format(FLAGS.name, config_name)
        )

    logger.info('Writing kinematics to {}'.format(FLAGS.output))
    jgnn.utils.dataset.write_graph_dataset(
        FLAGS.output, node_features, graph_features, num_stars,
        headers=headers)

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    logger.info('Time taken: {:.2f} seconds'.format(t2-t1))
