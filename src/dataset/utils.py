import torch

def nonlinearity(dataset, epsilon=1.):
    X = dataset[:]
    _, s, _ = torch.svd(X, compute_uv=False)
    return s.max() / s.min(), s.min() / epsilon