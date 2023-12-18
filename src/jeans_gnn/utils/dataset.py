
import logging
import os

logger = logging.getLogger(__name__)

import h5py
import numpy as np
from torch_geometric.loader import DataLoader


def write_graph_dataset(
        path, node_features, graph_features, lengths, headers=None):
    """ Write a graph dataset with node features and graph features into HDF5
    file. The node features of all graphs are concatenated into a single
    array. The lengths of each graph is required to split the node features
    into individual graphs.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    node_features : dict
        Dictionary of node features. The node features of all graphs are
        concatenated into a single array. The key is the name of the feature and
        the value is a list of arrays of shape (M, ) where M is the number
        of nodes in all graphs.
    graph_features : dict
        Dictionary of graph features. The key is the name of the feature and
        the value is a list of arrays of shape (N, ) where N is the number
        of graphs.
    lengths : list
        List of lengths of each graph. The length of the list is the number of
        graphs. The sum of the lengths must be equal to M, or the total number
        of nodes in all graphs.
    headers : dict
        Dictionary of headers. The key is the name of the header and the value
        is the value of the header.
    Returns
    -------
    None
    """
    # check if total length of node features is equal to the sum of lengths
    assert np.sum(lengths) == len(list(node_features.values())[0])

    # construct pointer to each graph
    ptr = np.cumsum(lengths)

    # define headers
    default_headers ={
        "node_features": list(node_features.keys()),
        "graph_features": list(graph_features.keys()),
    }
    default_headers['all_features'] = (
        default_headers['node_features'] + default_headers['graph_features'])
    if headers is None:
        headers = default_headers
    else:
        headers.update(default_headers)

    # write dataset into HDF5 file
    with h5py.File(path, 'w') as f:
        # write pointers
        f.create_dataset('ptr', data=ptr)

        # write node features
        for key in node_features:
            dset = f.create_dataset(key, data=node_features[key])
            dset.attrs.update({'type': 'node_features'})

        # write tree features
        for key in graph_features:
            dset = f.create_dataset(key, data=graph_features[key])
            dset.attrs.update({'type': 'graph_features'})

        # write headers
        f.attrs.update(headers)


def read_graph_dataset(path, features_list=None, concat=False, to_array=True):
    """ Read graph dataset from path and return node features, graph
    features, and headers.

    Parameters
    ----------
    path : str
        Path to the HDF5 file.
    features_list : list
        List of features to read. If empty, all features will be read.
    concat : bool
        If True, the node features of all graphs will be concatenated into a
        single array. Otherwise, the node features will be returned as a list
        of arrays.
    to_array : bool
        If True, the node features will be returned as a numpy array of
        dtype='object'. Otherwise, the node features will be returned as a
        list of arrays. This option is only used when concat is False.

    Returns
    -------
    node_features : dict
        Dictionary of node features. The key is the name of the feature and
        the value is a list of arrays of shape (M, ) where M is the number
        of nodes in all graphs.
    graph_features : dict
        Dictionary of graph features. The key is the name of the feature and
        the value is a list of arrays of shape (N, ) where N is the number
        of graphs.
    headers : dict
        Dictionary of headers.
    """
    if features_list is None:
        features_list = []

    # read dataset from HDF5 file
    with h5py.File(path, 'r') as f:
        # read dataset attributes as headers
        headers = dict(f.attrs)

        # if features_list is empty, read all features
        if len(features_list) == 0:
            features_list = headers['all_features']

        # read node features
        node_features = {}
        for key in headers['node_features']:
            if key in features_list:
                if f.get(key) is None:
                    logger.warning(f"Feature {key} not found in {path}")
                    continue
                if concat:
                    node_features[key] = f[key][:]
                else:
                    node_features[key] = np.split(f[key][:], f['ptr'][:-1])

        # read graph features
        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                if f.get(key) is None:
                    logger.warning(f"Feature {key} not found in {path}")
                    continue
                graph_features[key] = f[key][:]

    # convert node features to numpy array of dtype='object'
    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}
    return node_features, graph_features, headers

def create_dataloader_from_path(
        dataset_path, preprocess, verbose=True, **kwargs):
    """ Create a data loader from a dataset
    Parameters
    ----------
    dataset_path: str
        Path to the dataset
    preprocess: callable
        Transform function from coordinates to torch_geometric.data.Data
    verbose: bool
        Whether to print out the dataset information
    kwargs: dict
        Keyword arguments for DataLoader

    Returns
    -------
    dataloader: torch_geometric.loader.DataLoader
    """
    # find dataset path, return None if not found
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # read the dataset
    features_list = ['pos', 'vel', 'vel_error', 'labels']
    node_features, graph_features, headers = read_graph_dataset(
        dataset_path, features_list=features_list)
    pos = node_features['pos']
    vel = node_features['vel']
    # if the dataset does not contain velocity error
    vel_error = node_features.get('vel_error')

    # print out dataset information
    if verbose:
        logger.info(f"Dataset: {dataset_path}")
        logger.info(f"Number of graphs: {len(node_features['pos'])}")
        logger.info("Headers:")
        for header in headers:
            logger.info(f"{header}: {headers[header]}")

    # return dataloader
    return create_dataloader_from_array(
        pos, vel, preprocess, vel_error=vel_error, labels=graph_features['labels'],
        verbose=verbose, **kwargs)

def create_dataloader_from_array(
        pos, vel, preprocess, vel_error=None, labels=None, verbose=True, **kwargs):
    """ Create a data loader from a dataset
    Parameters
    ----------
    pos: np.ndarray
        Array of shape (N, 2) containing the positions of N particles
    vel: np.ndarray
        Array of shape (N, 1) containing the velocities of N particles
    preprocess: callable
        Preprocess function from coordinates to torch_geometric.data.Data
    vel_error: np.ndarray
        Array of shape (N, 1) containing the velocity errors of N particles
        If None, assume to be zero
    labels: np.ndarray
        Array of shape (N, ) containing the labels of N particles
    verbose: bool
    kwargs: dict
        Keyword arguments for DataLoader

    Returns
    -------
    dataloader: torch_geometric.loader.DataLoader
    """
    # create a graph dataset
    dataset = []
    if verbose:
        logger.info(f"Creating graph dataset")

    for i in range(len(pos)):
        # print every 10%
        if verbose:
            if len(pos) < 10:
                logger.info(f"Creating graph {i} / {len(pos)}")
            elif i % (len(pos) // 10) == 0:
                logger.info(f"Creating graph {i} / {len(pos)}")

        # create a graph
        label_i = None if labels is None else labels[i]
        if vel_error is None:
            vel_error_i = np.zeros_like(vel[i])
        else:
            vel_error_i = vel_error[i]

        graph = preprocess(pos[i], vel[i], vel_error_i, label=label_i)
        dataset.append(graph)

    # create a data loader
    return DataLoader(dataset, **kwargs)
