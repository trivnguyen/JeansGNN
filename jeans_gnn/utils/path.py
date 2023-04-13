
import os
from .. import envs

def find_dataset(name):
    """ Find the dataset directory

    Returns
    -------
    path : str
        Path to the dataset.
    """
    path = os.path.join(envs.DEFAULT_DATASETS_DIR, name)
    # return None if dataset not found
    if not os.path.exists(path):
        return None
        # raise FileNotFoundError("Dataset not found: {}".format(path))
    return path

def find_galaxy(name):
    """ Find the galaxy path.

    Returns
    -------
    path : str
        Path to the galaxy.
    """
    path = os.path.join(envs.DEFAULT_GALAXIES_DIR, name + '.hdf5')
    # return None if galaxy not found
    if not os.path.exists(path):
        return None
        raise FileNotFoundError("Galaxy not found: {}".format(path))
    return path