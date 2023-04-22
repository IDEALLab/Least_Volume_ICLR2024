import os, json
import torch
from torch.optim import Adam
import numpy as np
import argparse
from model.autoencoder import AutoEncoder
from model.sparsity import *
from model.cmpnts import DCDiscriminator, TrueSNDCGenerator, DCGenerator
from dataset.images import ImageToyDataset, MNISTImages, CIFAR10Images
from dataset.toy import TensorDataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.multiprocessing as mp

class Experiment:
    def __init__(self, configs, Encoder, Decoder, Optimizer, AE, device='cpu') -> None:
        self.configs = configs
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Optimizer = Optimizer
        self.AE = AE
        self.device = device
    
    def run(self, dataloader, epochs, lam, save_dir, eps=0):
        self.init_model(lam)
        epoch = self.model.fit(
            dataloader=dataloader,
            epochs=epochs, # maybe convergence criterion is better
            tb_writer=self.make_writer(save_dir),
            save_dir=save_dir, 
            save_iter_list=list(np.linspace(epochs//5, epochs, 5, dtype=int) - 1),
            eps=eps if eps > 0 else None
            )
        self.save_result(epoch if epoch is not None else epochs, save_dir)
    
    def save_result(self, epochs, save_dir):
        self.model.save(save_dir, epochs)
        
    def init_model(self, lam):
        self.model = self.AE(
            self.configs, self.Encoder, self.Decoder, 
            self.Optimizer, weights=[1., lam]
            )
        self.model.to(self.device)
    
    def make_writer(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        return SummaryWriter(save_dir)

def read_configs(name):
    with open(os.path.join('configs', name+'.json')) as f:
        configs = json.load(f)
    return configs

def load_dataset(name, size=(128, 128), device='cpu'):
    if name == 'sythetic':
        dataset = ImageToyDataset('../data/synthetic/images_30k.npy', size=size, device=device)
    elif name == 'mnist':
        dataset = MNISTImages(device=device)
    elif name == 'cifar10':
        dataset = CIFAR10Images(device=device)
    else:
        raise NotImplementedError('Dataset not supported.')
    return dataset

def main(name, ae_name, epochs=10000, batch=100, lam=1e-3, device='cpu', eps=0):
    dataset = load_dataset(name, device=device)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    configs = read_configs(name)
    save_dir = '../saves/image/{}/'.format(name)

    if ae_name == 'dp':
        experiment = Experiment(configs, DCDiscriminator, TrueSNDCGenerator, Adam, DynamicPruningAE_v2, device=device) # SNMLP for spectral normalization
    else:
        configs['name'] = 'non'
        experiment = Experiment(configs, DCDiscriminator, TrueSNDCGenerator, Adam, AutoEncoder, device=device)
    experiment.run(dataloader=dataloader, epochs=epochs, lam=lam, save_dir=save_dir, eps=eps) # epochs to be modified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='name of the dataset')
    parser.add_argument('-n', '--name', default='dp', help='name of the autoencoder')
    parser.add_argument('-d', '--device', default='cpu', help='device to run the experiments')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='number of samples in a mini-batch')
    parser.add_argument('-l', '--lam', type=float, default=1e-4, help='weight for least volume')
    parser.add_argument('--eps', type=float, default=0, help='covergence threshold')
    args = parser.parse_args()

    main(args.dataset, args.name, args.epochs, args.batch, args.lam, args.device, args.eps)