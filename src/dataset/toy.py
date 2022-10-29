import torch

def gen1d_in_2d(N=50):
    X, _ = torch.sort(torch.rand(N))
    X = X - 0.5
    Y = X + 5 * X ** 3
    data = torch.stack([X, Y], dim=-1)
    return data

