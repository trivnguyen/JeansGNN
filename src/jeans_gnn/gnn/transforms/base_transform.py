
from typing import List

class BaseTransform():
    """ Abstract base class for transformations """
    def __call__(self, batch):
        raise NotImplementedError

    def recompute_indim(self, indim: int) -> int:
        return indim

    def recompute_outdim(self, outdim: int) -> int:
        return outdim


class CompositeTransform(BaseTransform):
    """ Composite transformation """
    def __init__(self, transforms: List[BaseTransform]):
        self.transforms = transforms

    def __call__(self, batch):
        for transform in self.transforms:
            batch = transform(batch)
        return batch

    def recompute_indim(self, indim: int) -> int:
        for transform in self.transforms:
            indim = transform.recompute_indim(indim)
        return indim

    def recompute_outdim(self, outdim: int) -> int:
        for transform in self.transforms:
            outdim = transform.recompute_outdim(outdim)
        return outdim
