import os
import torch
from torch.optim import Adam
import numpy as np
import argparse
from model.autoencoder import AutoEncoder
from model.sparsity import *
from model.cmpnts import MLP, SNMLP
from dataset.toy import TensorDataset, DataLoader
import torch.multiprocessing as mp

class Experiment:
    def __init__(self, configs, Encoder, Decoder, Optimizer, AE, device='cpu') -> None:
        self.configs = configs
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Optimizer = Optimizer
        self.AE = AE
        self.device = device
    
    def run(self, dataloader, epochs, lams, save_dir, eps=0):
        for lam in lams:
            history = []
            self.init_model(lam)
            epoch = self.model.fit(
                dataloader=dataloader,
                epochs=epochs, # maybe convergence criterion is better
                history=history,
                eps=eps if eps > 0 else None
                )
            self.save_result(lam, history, epoch if epoch is not None else epochs, save_dir)
    
    def save_result(self, lam, history, epochs, save_dir):
        path = os.path.join(save_dir, '{:.0e}'.format(lam)) # model/man/amb/i/lam
        os.makedirs(path, exist_ok=True)
        self.model.save(path, epochs)
        np.savetxt(os.path.join(path, self.model.name+'_history.csv'), np.asarray(history))
        
    def init_model(self, lam):
        self.model = self.AE(
            self.configs, self.Encoder, self.Decoder, 
            self.Optimizer, weights=[1., lam]
            )
        self.model.to(self.device)

ae_dict = {
    'vol': VolumeAE,
    'l1': L1AE,
    'l2': L2AE,
    'lasso': LassoAE,
    'dp': DynamicPruningAE_v2,
    'non': AutoEncoder
}

##### Functions ######

def generate_configs(data_dim, width, name, lr=1e-4):
    configs = dict()
    configs['encoder'] = {
        'in_features': data_dim,
        'out_features': min(256, data_dim),
        'layer_width': width
        }
    configs['decoder'] = {
        'in_features': min(256, data_dim),
        'out_features': data_dim,
        'layer_width': width
        }
    configs['optimizer'] = {'lr': lr}
    configs['name'] = name
    configs['data_dim'] = configs['encoder']['in_features']
    configs['beta'] = 0.9
    configs['z_t'] = 1e-2
    configs['r_t'] = 1e-2
    return configs

def read_dataset(latent_dim, data_dim, i, device='cpu'):
    path = '../data/toy_new/{}-manifold/{}-ambient/'.format(latent_dim, data_dim)
    data_name = '{}-{}_{}.npy'.format(latent_dim, data_dim, i)
    tensor = torch.as_tensor(np.load(os.path.join(path, data_name)), dtype=torch.float, device=device) # type:ignore
    return TensorDataset(tensor)

def create_savedir(l, d, i):
    dir = os.path.join('../saves/toy_new/{}-man/{}-amb/#{}/'.format(l, d, i))
    os.makedirs(dir, exist_ok=True)
    return dir

def main_mp(ae_name, i, epochs=10000, batch=100, device='cpu', eps=0):
    ll = [1, 2, 4, 8, 16, 32][:4]
    dd = [2, 4, 8, 16, 32]
    lams = 10 ** np.linspace(-6, 0, 7) # change to [0] for 'non'
    ww = [[32]*4, [64]*4, [128]*4, [256]*4, [512]*4, [1024]*4][:4]
    recs = np.load('../data/toy_new/recs.npy')
    for j, (l, width) in enumerate(zip(ll, ww)):
        for k, d in enumerate(dd):
            dataset = read_dataset(l, d*l, i, device=device)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
            configs = generate_configs(d*l, width, ae_name)
            configs['r_t'] = recs[j, k, i] * 3
            save_dir = create_savedir(l, d*l, i)

            experiment = Experiment(configs, MLP, SNMLP, Adam, ae_dict[ae_name], device=device) # SNMLP for spectral normalization
            experiment.run(dataloader=dataloader, epochs=epochs, lams=lams, save_dir=save_dir, eps=eps) # epochs to be modified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ae_name', help='name of the autoencoder to be trained')
    parser.add_argument('-d', '--device', default='cpu', help='device to run the experiments')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='number of samples in a mini-batch')
    parser.add_argument('-i', '--folds', type=int, default=4, help='number of cross validation folds')
    parser.add_argument('--eps', type=float, default=0, help='covergence threshold')
    args = parser.parse_args()

    # main(args.ae_name, args.epochs, args.batch, args.device)
    for i in range(args.folds):
        p = mp.Process(target=main_mp, args=(args.ae_name, i, args.epochs, args.batch, args.device, args.eps))
        p.start()