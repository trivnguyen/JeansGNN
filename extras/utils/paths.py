
import os
from .. import envs

def find_dataset(name, flag=None):
    """ Find the dataset directory
    Parameters
    ----------
    name : str
        Name of the dataset.
    flag : str
        Flag of the dataset. Either 'train', 'val', or 'test'.
        If None, return the dataset directory.

    Returns
    -------
    path : str
        Path to the dataset. If flag is None, return the dataset directory.
    """
    path = os.path.join(envs.DEFAULT_DATASETS_DIR, name)
    if not os.path.exists(path):
        return None
    if flag is None:
        return path
    else:
        path = os.path.join(path, flag + '.hdf5')
        if not os.path.exists(path):
            return None
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
    return path