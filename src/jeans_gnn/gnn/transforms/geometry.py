
from typing import List

import torch
from .base_transform import BaseTransform

def project2d(pos, vel, axis=0, use_proper_motions=False):
    """ Project the 3D positions and velocities to 2D.
    Return the 2d positions and line-of-sight velocities.

    Parameters
    ----------
    pos : array_like
        3D positions shape (N, 3)
    vel : array_like
        3D velocities shape (N, 3)
    axis : int or str
        The LOS Axis to project to (0, 1, or 2). If None, apply a random projection.
    use_proper_motions : bool
        Whether to include proper motions in the velocities

    Returns
    -------
    pos_proj : array_like
        2D positions
    vel_proj: array_like
        Line-of-sight velocities
    """
    # if axis is None, apply a random projection
    # by randomly rotate the 3D positions and velocities
    if axis == 'random':
        # random rotation matrix
        R = np.random.randn(3, 3)
        R = np.dot(R, R.T)
        pos = np.dot(pos, R)
        vel = np.dot(vel, R)
        axis = np.random.randint(3)
    # project to 2D
    pos_proj = np.delete(pos, axis, axis=1)

    if use_proper_motions:
        return pos_proj, vel
    else:
        return pos_proj, vel[:, axis]


class Project(BaseTransform):
    """ Project the input features onto the specified axis """
    def __init__(self):
        """
        Parameters
        ----------
        """
        super().__init__()

    def __call__(self, batch):
        """ Project the input features onto the specified axis """
        batch = batch.clone()
        with torch.no_grad():
            # project the input features onto the specified axis
            batch.x = batch.x[:, 0:1]
        return batch
