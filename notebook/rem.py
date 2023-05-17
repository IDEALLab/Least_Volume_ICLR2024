# %%
import torch
import numpy as np
import sys; sys.path.append('../src/'); sys.path.append('..')
import matplotlib.pyplot as plt
import glob

# %%
from src.model.utils.metrics import l2_loss, explained_reconstruction, mean_correlation, importance_correlation, main_exprec
from src.least_volume_image import *
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader
from math import sqrt

# %%
ae_dic = {
    'vol': VolumeAE_BCE,
    'l1': L1AE_BCE,
    'lasso': LassoAE_BCE,
    'bce': BCEAutoencoder
}

def load_model(ae_name, json_dir, tar_dir, lam, lip=True, device='cpu'):
    with open(json_dir) as f: configs = json.load(f)
    AE = ae_dic[ae_name]
    Decoder = TrueSNDCGeneratorSig if lip else DCGeneratorSig
    model = AE(configs, DCDiscriminator, Decoder, Adam, weights=[1., lam]).to(device)
    model.load(tar_dir)
    model.eval()
    return model

def get_dataset(name, train=True, device='cpu'):
    dataset, _ = load_dataset(name, train=train, device=device)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    return next(iter(dataloader))

@torch.no_grad()
def get_metrics(model, dataset, metrics):
    return [metric(model, dataset) for metric in metrics]

def ae_statistics(data_name, ae_name, group, epoch, lams, metrics, eps=None, lip=True, comment='', train=True, device='cpu', src='../saves/image/'):
    if not lip: comment = comment + '_nolip'
    if ae_name == 'vol' and eps is not None: comment = '_e{}'.format(eps) + comment

    dataset = get_dataset(data_name, train=train, device=device)
    stats = []
    for lam in lams:
        dir = os.path.join(src, data_name, group, '{}_{}{}/'.format(ae_name, lam, comment))
        print(dir)
        json_file = glob.glob('*.json', root_dir=dir)[0]
        json_dir = os.path.join(dir, json_file)
        tar_file = glob.glob('*{}.tar'.format(epoch), root_dir=dir)[0]
        tar_dir = os.path.join(dir, tar_file)

        model = load_model(ae_name, json_dir, tar_dir, lam, lip, device)
        stats.append(get_metrics(model, dataset, metrics))
    return stats


# %%
from tqdm import tqdm, trange

def prune(k, z, descending):
    std, idx = z.std(0).sort(descending=descending)
    mean = z.mean(0)
    z[:, idx[:k]] = mean[idx[:k]]
    return z

def l2_prune(k=0, descending=True):
    def _l2_(model, dataset):
        z = model.encode(dataset)
        z = prune(k, z, descending)
        rec = model.decode(z)
        return l2_loss(dataset, rec)
    return _l2_

def l2_ps(model, dataset):
    z = model.encode(dataset)
    std, idx = z.std(0).sort(descending=True)
    mean = z.mean(0)
    l2s = []
    for i in tqdm(idx):
        z_ = z.clone()
        z_[:, i] = mean[i]
        rec = model.decode(z_)
        l2s.append(l2_loss(dataset, rec))
    return torch.stack(l2s)

def l2_cum(descending=True):
    def _l2_(model, dataset):
        l2s = []
        z = model.encode(dataset)
        for i in trange(z.size(1)):
            _l2 = l2_prune(k=i+1, descending=descending)
            l2s.append(_l2(model, dataset))
        return torch.stack(l2s)
    return _l2_

def z_std(model, dataset):
    z = model.encode(dataset)
    std, idx = z.std(0).sort(descending=True)
    return std

def z_index(model, dataset):
    z = model.encode(dataset)
    std, idx = z.std(0).sort(descending=True)
    return idx


import torch.multiprocessing as mp

metrics = [z_std] #[l2_prune(0), l2_prune(None), l2_ps, l2_cum(True), l2_cum(False), z_index]
names = ['z_std'] # ['l2_non', 'l2_all', 'l2_each', 'l2_cum_d', 'l2_cum_a', 'z_index']

group = 'vol'
ae_name = 'vol'

def vol_main(device, eps):
    for dataset_name, lams, epoch in zip(['syn', 'mnist', 'cifar10'], \
                                        [(1e-2, 3e-3, 1e-3, 3e-4, 1e-4), \
                                        (3e-2, 1e-2, 3e-3, 1e-3, 3e-4), \
                                        (3e-2, 1e-2, 3e-3, 1e-3, 3e-4)], \
                                        [399, 399, 999]):
        stats = ae_statistics(dataset_name, ae_name, group=group, epoch=epoch, lams=lams, eps=eps, metrics=metrics, device='cuda:{}'.format(device))

        path = os.path.join('../saves/image/', dataset_name, group)
        for i, nm in enumerate(names):
            ls = []
            for each in stats:
                ls.append(each[i])
            np.save(os.path.join(path, 'e{}_{}.npy'.format(eps, nm)), torch.stack(ls).cpu().numpy())

# %%
for i, eps in zip(range(5), [0., 1., 3., 10., 30.]):
    p = mp.Process(target=vol_main, args=(i, eps))
    p.start()


