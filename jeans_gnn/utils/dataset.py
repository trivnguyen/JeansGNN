
import h5py
import numpy as np

def write_graph_dataset(
        path, node_features, graph_features, lengths, headers={}):
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


def read_graph_dataset(path, features_list=[], concat=False, to_array=True):
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
                if concat:
                    node_features[key] = f[key][:]
                else:
                    node_features[key] = np.split(f[key][:], f['ptr'][:-1])

        # read graph features
        graph_features = {}
        for key in headers['graph_features']:
            if key in features_list:
                graph_features[key] = f[key][:]

    # convert node features to numpy array of dtype='object'
    if not concat and to_array:
        node_features = {
            p: np.array(v, dtype='object') for p, v in node_features.items()}
    return node_features, graph_features, headers
