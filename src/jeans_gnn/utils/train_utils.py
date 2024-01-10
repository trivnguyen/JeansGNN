
import torch

def pad_and_create_mask(features, max_len=None):
    """ Pad and create Transformer mask. """
    if max_len is None:
        max_len = max([f.shape[0] for f in features])

    # create mask (batch_size, max_len)
    # note that jax mask is 1 for valid entries and 0 for invalid entries
    # this is the opposite of the pytorch mask
    mask = torch.zeros((len(features), max_len), dtype=torch.bool)
    for i, f in enumerate(features):
        mask[i, f.shape[0]:] = True

    # zero pad features
    padded_features = torch.zeros((len(features), max_len, features[0].shape[1]))
    for i, f in enumerate(features):
        padded_features[i, :f.shape[0]] = f

    return padded_features, mask