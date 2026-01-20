import torch
from torch.nn import functional as F


def masks_sample_points(masks: torch.Tensor, k: int = 10):
    samples = []
    for batch_index in range(masks.shape[0]):
        mask = masks[batch_index]
        pos_y_indices, pos_x_indices = mask.nonzero(as_tuple=True)

        perm = torch.randperm(pos_x_indices.shape[0])
        index = perm[:k]

        samples_x = pos_x_indices[index]
        samples_y = pos_y_indices[index]

        samples_xy = torch.stack([samples_x, samples_y], dim=1)

        samples.append(samples_xy)

    samples = torch.stack(samples, dim=0)

    return samples
