import os, json
import torch
from torch.optim import Adam
import numpy as np
import argparse
from model.autoencoder import AutoEncoder, BCEAutoencoder
from model.sparsity import *
from model.cmpnts import DCDiscriminator, TrueSNDCGenerator, DCGenerator, DCGeneratorSig, TrueSNDCGeneratorSig
from dataset.images import *
from dataset.toy import TensorDataset, DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import torch.multiprocessing as mp
from shutil import copyfile

class Experiment:
    def __init__(self, configs, Encoder, Decoder, Optimizer, AE, device='cpu') -> None:
        self.configs = configs
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.Optimizer = Optimizer
        self.AE = AE
        self.device = device
    
    def run(self, dataloader, epochs, lam, save_dir, save_num=10, eps=0):
        self.init_model(lam)
        epoch = self.model.fit(
            dataloader=dataloader,
            epochs=epochs, # maybe convergence criterion is better
            tb_writer=self.make_writer(save_dir),
            save_dir=save_dir, 
            save_iter_list=list(np.linspace(epochs//save_num, epochs, save_num, dtype=int) - 1),
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

def read_configs(name, src='./configs/'):
    with open(os.path.join(src, name+'.json')) as f:
        configs = json.load(f)
    return configs

def load_dataset(name, train=True, device='cpu'):
    if 'syn' in name:
        dataset = ImageToyDataset('../data/synthetic/images_30k.npy', size=(32, 32), device=device)
        id = 'syn'
    elif 'mnist' in name:
        dataset = MNISTImages(train=train, device=device)
        id = 'mnist'
    elif 'cifar10' in name:
        dataset = CIFAR10Images(train=train, device=device)
        id = 'cifar10'
    elif 'celeba' in name:
        dataset = CelebAImages(train=train, device=device)
        id = 'celeba'
    else:
        raise NotImplementedError('Dataset not supported.')
    return dataset, id

def main(name, ae_name, epochs=10000, batch=100, lam=1e-4, sigmoid=True, device='cpu', save_num=10, nolip=False, eps=1, comment=''):
    dataset, id = load_dataset(name, device=device)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    configs = read_configs(name)
    if nolip: comment = comment + '_nolip'
    if ae_name == 'vol': comment = '_e{}'.format(eps) + comment
    save_dir = '../saves/image/{}/{}/{}_{}{}/'.format(id, ae_name, ae_name, lam, comment)
    os.makedirs(save_dir, exist_ok=True)
    copyfile('./configs/{}.json'.format(name), os.path.join(save_dir, '{}.json'.format(name)))

    Decoder = (TrueSNDCGeneratorSig if sigmoid else TrueSNDCGenerator) if not nolip else (DCGeneratorSig if sigmoid else DCGenerator)

    if ae_name == 'dp':
        experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, DynamicPruningAE_BCEO, device=device) # SNMLP for spectral normalization
    # elif ae_name == 'dpr':
    #     experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, DynamicPruningAE_BCE, device=device)
    elif ae_name == 'vol':
        configs['eps'] = eps
        experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, VolumeAE_BCE, device=device)
    # elif ae_name == 'vol_new':
    #     experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, DynamicPruningAE_BCEv2, device=device) # DynamicPruningAE_BCEv2
    # elif ae_name == 'l1_new':
    #     experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, DynamicPruningAE_BCEv3, device=device) # DynamicPruningAE_BCEv2
    elif ae_name == 'l1':
        experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, L1AE_BCE, device=device)
    # elif ae_name == 'l1v':
    #     experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, L1AE_BCEv, device=device) 
    # elif ae_name == 'l1dp':
    #     experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, L1AE_BCE_dp, device=device)
    elif ae_name == 'lasso':
        experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, LassoAE_BCE, device=device) 
    elif ae_name == 'bce':
        experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, BCEAutoencoder, device=device)
    else:
        configs['name'] = 'non'
        experiment = Experiment(configs, DCDiscriminator, Decoder, Adam, AutoEncoder, device=device)
    experiment.run(dataloader=dataloader, epochs=epochs, lam=lam, save_dir=save_dir, save_num=save_num) # epochs to be modified


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='name of the dataset')
    parser.add_argument('-n', '--name', default='dp', help='name of the autoencoder')
    parser.add_argument('-d', '--device', default='cpu', help='device to run the experiments')
    parser.add_argument('-e', '--epochs', type=int, default=10000, help='number of epochs')
    parser.add_argument('-b', '--batch', type=int, default=100, help='number of samples in a mini-batch')
    parser.add_argument('-l', '--lam', type=float, default=1e-4, help='weight for least volume')
    parser.add_argument('-s', '--sig', type=bool, default=True, help='sigmoid for decoder')
    parser.add_argument('--nolip', action='store_true', help='no Lipschitz regularization')
    parser.add_argument('--eps', type=float, default=1, help='covergence threshold')
    parser.add_argument('--num', type=int, default=10, help='number of saves')
    parser.add_argument('--com', type=str, default='', help='comment')
    args = parser.parse_args()

    main(args.dataset, args.name, args.epochs, args.batch, args.lam, args.sig, args.device, args.num, args.nolip, args.eps, args.com)