import numpy as np
import torch
import cv2

def mixup_data(x, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lamb = np.random.beta(alpha + 1., alpha)
    else:
        lamb = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lamb * x + (1 - lamb) * x[index, :]
    return mixed_x


def cutmix_data(inputs, alpha=1.):
    """
    Returns cut-mixed inputs, pairs of targets, and lambda.
    """
    bsize, _, h, w = inputs.shape
    shuffled_idxs = torch.randperm(bsize).cuda()

    inputs_s = inputs[shuffled_idxs]
    lamb = np.random.beta(alpha + 1., alpha)

    rx = np.random.randint(w)
    ry = np.random.randint(h)
    cut_ratio = np.sqrt(1. - lamb)
    rw = np.int(cut_ratio * w)
    rh = np.int(cut_ratio * h)

    x1 = np.clip(rx - rw // 2, 0, w)
    x2 = np.clip(rx + rw // 2, 0, w)
    y1 = np.clip(ry - rh // 2, 0, h)
    y2 = np.clip(ry + rh // 2, 0, h)

    inputs[:, :, x1:x2, y1:y2] = inputs_s[:, :, x1:x2, y1:y2]
    return inputs
