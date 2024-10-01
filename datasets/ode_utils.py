import torch
import numpy as np

def _numerical_diff(trajectory, times):
    y_grad = torch.gradient(trajectory, spacing=(times,), dim=0)[0]
    return y_grad


def to_torch(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array).float()
    if isinstance(array, torch.Tensor):
        return array