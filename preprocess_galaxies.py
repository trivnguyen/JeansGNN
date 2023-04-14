
# import modules
import argparse
import datetime
import logging
import os
import time

import astropy.units as u
import numpy as np
import pandas as pd
import yaml

from jeans_gnn import utils, envs

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
        '--galaxy-name', type=str, required=True,
        help='Name of input galaxy to preprocess')
    parser.add_argument(
        '--dataset-name', type=str, required=False,
        help='Name of the output dataset. If not specified, use the galaxy name.')

    args = parser.parse_args()
    if args.dataset_name is None:
        args.dataset_name = args.galaxy_name

    return args

def get_graph(node_features, graph_features, idx):
    """ Get a graph from the dataset given the index """
    nodes = {}
    for k, v in node_features.items():
        nodes[k] = v[idx]
    graph = {}
    for k, v in graph_features.items():
        graph[k] = v[idx]
    return nodes, graph


def project2d(pos, vel, axis=0):
    """ Project the 3D positions and velocities to 2D.
    Return the 2d positions and line-of-sight velocities.

    Parameters
    ----------
    pos : array_like
        3D positions shape (N, 3)
    vel : array_like
        3D velocities shape (N, 3)
    axis : int or None
        Axis to project to (0, 1, or 2). If None, apply a random projection.

    Returns
    -------
    pos2d : array_like
        2D positions
    vel_los: array_like
        Line-of-sight velocities
    """
    # if axis is None, apply a random projection
    # by randomly rotate the 3D positions and velocities
    if axis is None:
        # random rotation matrix
        R = np.random.randn(3, 3)
        R = np.dot(R, R.T)
        pos = np.dot(pos, R)
        vel = np.dot(vel, R)
        axis = np.random.randint(3)
    # project to 2D
    pos2d = np.delete(pos, axis, axis=1)
    vel_los = vel[:, axis]
    return pos2d, vel_los


def parse_graph_features(graph_features):
    """ Parse graph features into training target """

    # create a copy of the graph features
    new_graph_features = graph_features.copy()

    # parse DM parameters
    new_graph_features['dm_log_r_dm'] = np.log10(graph_features['dm_r_dm'])
    new_graph_features['dm_log_rho_0'] = np.log10(graph_features['dm_rho_0'])

    # parse stellar parameters
    if graph_features.get('stellar_r_star') is not None:
        new_graph_features['stellar_log_r_star'] = np.log10(
            graph_features['stellar_r_star'])
    elif graph_features.get('stellar_r_star_r_dm') is not None:
        new_graph_features['stellar_log_r_star'] = (
            np.log10(graph_features['stellar_r_star_r_dm']) + new_graph_features['dm_log_r_dm'])
    else:
        raise ValueError('Cannot find stellar radius')

    # parse DF parameters
    if graph_features.get('df_r_a') is not None:
        new_graph_features['df_log_r_a'] = np.log10(graph_features['df_r_a'])
    elif graph_features.get('df_r_a_r_dm') is not None:
        new_graph_features['df_log_r_a'] = (
            np.log10(graph_features['df_r_a_r_dm']) + new_graph_features['dm_log_r_dm'])
    elif graph_features.get('df_r_a_r_star') is not None:
        new_graph_features['df_log_r_a'] = (
            np.log10(graph_features['df_r_a_r_star']) + new_graph_features['stellar_log_r_star'])
    else:
        raise ValueError('Cannot find DF scale radius')

    return new_graph_features


def main():
    """ Preprocess galaxies into dataset """
    FLAGS = parse_args()

    # Load config file and parameters
    logger.info('Loading config file {}'.format(FLAGS.config))
    with open(FLAGS.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    projection  =config['preprocess'].get('projection')
    error_los = config['preprocess'].get('error_los', 0)
    labels_order = config['preprocess']['labels_order']
    train_frac = config['preprocess']['train_frac']

    # Find and read galaxies as a graph dataset
    node_features, graph_features, headers = utils.datasets.read_graph_datasets(
        utils.paths.find_galaxies(FLAGS.galaxy_name), to_array=True)
    num_galaxies = headers['num_galaxies']

    # Create a new graph dataset with node and graph features
    # Preprocess node features
    ppr_node_features = {
        'pos': [],
        'vel': [],
        'vel_error': [],
    }

    for i in range(num_galaxies):
        # extract galaxy
        nodes, graph = get_graph(node_features, graph_features, i)
        pos = nodes['pos']
        vel = nodes['vel']

        # project to 2D
        pos2d, vel_los = project2d(pos, vel, axis=projection)

        # add noise to velocities
        vel_error = np.random.normal(0, error_los, size=vel_los.shape)
        vel_los += vel_error

        # add to the new dataset
        ppr_node_features['pos'].append(pos2d)
        ppr_node_features['vel'].append(vel_los)
        ppr_node_features['vel_error'].append(vel_error)

    # Preprocess graph features
    ppr_graph_features = parse_graph_features(graph_features)
    for k in labels_order:
        if k not in ppr_graph_features:
            labels_order.remove(k)
    ppr_graph_features['labels'] = np.array(
        [ppr_graph_features[k] for k in labels_order]).T

    # Write dataset to disk
    # create default headers
    default_headers = headers.copy()
    default_headers.update({
        'projection': projection if projection is not None else 'random',
        'error_los': error_los,
        'labels_order': labels_order,
        'train_frac': train_frac,
        'galaxy_name': FLAGS.galaxy_name,
        'dataset_name': FLAGS.dataset_name
    })

    # create an output directory
    output_dir = os.path.join(
        envs.DEFAULT_DATASETS_DIR, FLAGS.dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # split the dataset into training and validation and write to disk
    # shuffle the dataset
    idx = np.arange(num_galaxies)
    np.random.shuffle(idx)
    num_stars = ppr_graph_features['num_stars']

    for flag in ['train', 'valid']:
        # split the dataset
        if flag == 'train':
            idx_split = idx[:int(train_frac * num_galaxies)]
        else:
            idx_split = idx[int(train_frac * num_galaxies):]

        flag_node_features = {
            k: np.take(v, idx_split)
            for k, v in ppr_node_features.items()
        }
        flag_graph_features = {
            k: np.take(v, idx_split)
            for k, v in ppr_graph_features.items()
        }
        flag_graph_features['original_idx'] = idx_split

        # create headers
        flag_headers = default_headers.copy()
        flag_headers.update({
            'num_galaxies': len(idx_split),
            'flag': flag,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        })
        # concatenate node features
        flag_node_features = {
            k: np.concatenate(v, axis=0) for k, v in flag_node_features.items()
        }

        # write to disk
        filename = os.path.join(output_dir, '{}.h5'.format(flag))
        utils.write_graph_dataset(
            filename, flag_node_features, flag_graph_features,
            num_stars[idx_split], flag_headers
        )

if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    logger.info('Time taken: {:.2f} seconds'.format(t2-t1))
