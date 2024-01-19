import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.multiprocessing as mp
import os
from model.cmpnts import MLP
from scipy.stats import ortho_group
import matplotlib.pyplot as plt
from math import sqrt
from dataset.toy import IsometricEmbedding
from model.flow import _RealNVP


data_dir = '../data/toy_new'

def nonlinearity(dataset, epsilon=1.): # epsilon can be set to the groundtruth LD manifold's average std.
    X = dataset[:]
    _, s, _ = torch.svd(X - X.mean(0), compute_uv=False)
    return s.max() / s.min(), s.min() / epsilon, epsilon, s

@ torch.no_grad()
def save_dataset(X, Y, i):
    in_dim = Y.in_dim
    out_dim = Y.out_dim
    path = os.path.join(data_dir, '{}-manifold'.format(in_dim), '{}-ambient'.format(out_dim))
    os.makedirs(path, exist_ok=True)
    
    data_path = os.path.join(path, '{}-{}_{}.npy'.format(in_dim, out_dim, i))
    report_path = os.path.join(path, '{}-{}_{}.txt'.format(in_dim, out_dim, i))
    np.save(data_path, Y[:].detach().cpu().numpy())
    
    _, s_x, _ = torch.svd(X - X.mean(), compute_uv=False)
    dis, ratio, eps, s = nonlinearity(Y, epsilon=s_x.mean())
    with open(report_path, 'w') as report:
        report.write('s.max / s.min = {}\n'.format(dis))
        report.write('s.min / x.s.mean = {}\n'.format(ratio))
        report.write('x.s.mean = {}\n'.format(eps / sqrt(len(X))))
        report.write('singular values = {}'.format(s.sort()[0].flip(0) / sqrt(len(X))))

def build_flow(out_dim, n=10, w=256):
    nets = lambda: nn.Sequential(
        nn.Linear(out_dim, w), nn.SiLU(), # SiLU
        nn.Linear(w, w), nn.SiLU(),
        nn.Linear(w, w), nn.SiLU(),
        nn.Linear(w, out_dim), nn.Tanh()) # must be tanh
    nett = lambda: nn.Sequential(
        nn.Linear(out_dim, w), nn.SiLU(), 
        nn.Linear(w, w), nn.SiLU(),
        nn.Linear(w, w), nn.SiLU(),
        nn.Linear(w, out_dim))
    masks = torch.as_tensor([
        [0]*(out_dim//2) + [1]*(out_dim//2), 
        [1]*(out_dim//2) + [0]*(out_dim//2)] * n, dtype=torch.float)
    flow = _RealNVP(nets, nett, masks)
    return flow


def run(i, device='cpu'):
    device = 'cuda'
    in_dims = [1, 2, 4, 8, 16, 32, 64][:4]
    out_dims = np.asarray([2, 4, 8, 16, 32], dtype=int)
    sizes = [300, 1000, 10000, 30000, 10000, 30000, 100000][:4]

    for in_dim, size in zip(in_dims, sizes):
        for out_dim in out_dims * in_dim:
            print('{} -> {} ({})'.format(in_dim, out_dim, i))

            X = torch.distributions.Uniform(
                -torch.ones(in_dim).to(device) / 2,
                torch.ones(in_dim).to(device) / 2
                )
            
            flow = build_flow(out_dim, 10, 256).to(device)
            Y = IsometricEmbedding(X, out_dim, flow, size)
            
            Y.embed(optimizer = torch.optim.Adam(Y.flow.parameters(), lr=1e-4),
                    epochs = int(100 * size / 300 * sqrt(out_dim // in_dim)),
                    batch_size = 100,
                    lam = 1.)

            save_dataset(X.sample(torch.Size([size])).detach().cpu(), Y, i)


if __name__ == '__main__':
    for i in range(4):
        p = mp.Process(target=run, args=(i,))
        p.start()