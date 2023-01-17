from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import functorch as ft
from torch.utils.data import Dataset, DataLoader
import sys; sys.path.append('../src/')
from model.cmpnts import MLP
from tqdm import tqdm, trange

class TensorDataset(Dataset):
    def __init__(self, tensor) -> None:
        super().__init__()
        self.tensor = tensor
    
    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]


class GraphDataset(Dataset):
    def __init__(
        self, 
        X: Union[torch.Tensor, Dataset],   # low dimensional flattened dataset of input manifold
        out_dim:int                     # dimension of target ambient space
        ):
        self.out_dim = out_dim
        self.in_dim = X[:].size(1)
        assert self.in_dim <= self.out_dim, 'Target data space has lower dimension.'
        self.dataset = self._embed(X[:])

    def _embed(self, X):
        fn_dim = self.out_dim-self.in_dim
        fn = MLP(self.in_dim, fn_dim, [32, 32])
        Y = fn(X)
        return torch.cat([X, Y], dim=-1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class IsometricEmbedding(Dataset):
    def __init__(
        self, 
        X: Union[torch.Tensor, Dataset],
        out_dim: int
        ):
        self.out_dim = out_dim
        self.in_dim = X[0].flatten().size(0)
        assert self.in_dim <= self.out_dim, 'Target data space has lower dimension.'
        self.X = X if isinstance(X, Dataset) else TensorDataset(X)

    def embed(
        self,
        flow: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int=500,
        batch_size: int=32,
        lam: float=1.
        ):
        flow.train()
        def fn(X):
            fn_dim = self.out_dim - self.in_dim
            X_ = torch.cat([X, torch.zeros(len(X), fn_dim, device=X.device)], dim=-1)
            return flow(X_)

        dataloader = DataLoader(self.X, batch_size, shuffle=True)
        iso_av = nlr_av = None # moving average
        with trange(epochs, 
                    bar_format='{l_bar}{bar:30}{r_bar}', 
                    desc='Embedding') as pbar:
            for _ in pbar:
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss_iso, loss_nlr = self._loss(batch, fn)
                    loss = lam * loss_iso + loss_nlr
                    loss.backward()
                    optimizer.step()

                    pbar.set_postfix({'isometricity': torch.sqrt(loss_iso).item(), 
                        'linearity': torch.sqrt(loss_nlr).item()}); pbar.refresh()
                    with torch.no_grad(): 
                        iso_av = loss_iso.detach() if iso_av is None else iso_av.lerp(loss_iso.detach(), 0.1) # avoid memory leak
                        nlr_av = loss_nlr.detach() if nlr_av is None else nlr_av.lerp(loss_nlr.detach(), 0.1)

        flow.eval()
        with torch.no_grad(): 
            self.dataset = fn(self.X[:])
            self.isometricity = torch.sqrt(iso_av.detach())   # type: ignore
            self.linearity = torch.sqrt(nlr_av.detach())   # type: ignore

    def _loss(self, X, f):
        V = F.normalize(torch.randn_like(X), dim=1)
        Y, P = ft.jvp(f, (X,), (V,))  # type: ignore
        
        norm = P.flatten(1).norm(dim=1)
        loss_iso = F.mse_loss(norm, torch.ones_like(norm))
        
        _, s, _ = torch.svd(Y - Y.mean(0), compute_uv=False)
        loss_nlr = s.var() / len(Y)
        return loss_iso, loss_nlr

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]