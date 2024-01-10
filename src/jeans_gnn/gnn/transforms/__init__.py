
from typing import Dict

from .base_transform import BaseTransform, CompositeTransform
from .noise_transform import UniformNoise, GaussianNoise
from .binary_stars_transform import AddBinaryPopulation

ALL_TRANSFORMS = {
    'UniformNoise': UniformNoise,
    'GaussianNoise': GaussianNoise,
    'AddBinaryPopulation': AddBinaryPopulation
}

def create_composite_transform(transform_dict: Dict) -> CompositeTransform:
    """ Create a composite transform from a dictionary """
    transforms = []
    num_transforms = transform_dict['num_transforms']
    for i in range(num_transforms):
        t_dict = transform_dict['transform_{:d}'.format(i)]
        t_type = t_dict['type']
        if t_type not in ALL_TRANSFORMS:
            raise ValueError("Unknown transform type: {:s}".format(t_type))
        transforms.append(ALL_TRANSFORMS[t_type](**t_dict['params']))
    return CompositeTransform(transforms)
