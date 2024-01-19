import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10, CelebA
from tqdm import trange

def generate_synthetic_dataset(
            img_dir: str, N: int, 
            degrees=90, 
            translate=(0.2, 0.2), 
            scale=(0.2, 0.5), 
            brightness=(0.3, 1), 
            hue=(0, 0.5)
            ):
        img = torchvision.io.read_image(img_dir)
        img = torch.cat([img, torch.zeros_like(img), torch.zeros_like(img)])
        tf = nn.Sequential(
            tt.RandomAffine(degrees=degrees, translate=translate, scale=scale),
            tt.ColorJitter(hue=hue, brightness=brightness),
            tt.Resize(32)
        )
        imgs = [tf(img) for _ in trange(N)]
        return torch.stack(imgs)


class ImageToyDataset(Dataset):
    def __init__(self, npy_dir: str, size=(32, 32), device='cpu') -> None: # h x w
        super().__init__()
        self.images = torch.as_tensor(np.load(npy_dir), dtype=torch.float)
        self.device = device
        self.resize(size)

    def resize(self, size):
        if self.size == size:
            pass
        else:
            self.images = F.interpolate(self.images, size)

    @property
    def size(self):
        return self.images.shape[-2:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].to(self.device)
    

class MNISTImages(Dataset):
    def __init__(self, train=True, device='cpu', preload=True) -> None: # h x w
        super().__init__()
        self.preload = preload
        if self.preload:
            self.images = torch.load('../data/mnist/mnist_trans.pt') \
                if train else torch.load('../data/mnist/mnist_trans_test.pt')
        else:
            self.images = MNIST(
                '../data/mnist/', train=train, download=True, 
                transform=torchvision.transforms.Compose(
                    [torchvision.transforms.Resize(32), torchvision.transforms.ToTensor()])
            )
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.preload:
            return self.images[idx].to(self.device)
        else:
            return self.images[idx][0].to(self.device)
    
class CIFAR10Images(Dataset):
    def __init__(self, train=True, device='cpu', preload=True) -> None: # h x w
        super().__init__()
        self.preload = preload
        if self.preload:
            self.images = torch.load('../data/cifar10/cifar10_trans.pt') \
                if train else torch.load('../data/cifar10/cifar10_trans_test.pt')
        else:
            self.images = CIFAR10(
            '../data/cifar10/', train=train, download=True, 
            transform=torchvision.transforms.ToTensor()
            )
        
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.preload:
            return self.images[idx].to(self.device)
        else:
            return self.images[idx][0].to(self.device)


class CelebAImages(Dataset):
    def __init__(self, train=True, device='cpu', preload=True) -> None: # h x w
        super().__init__()
        self.preload = preload
        if self.preload:
            self.images = torch.load('../data/celeba/celeba_trans.pt') \
                if train else torch.load('../data/celeba/celeba_trans_test.pt')
        else:
            self.images = CelebA(
            '../data/celeba/', split='train' if train else 'test', download=True, 
            transform=torchvision.transforms.Compose(
                [   
                    torchvision.transforms.CenterCrop(150),
                    torchvision.transforms.Resize(32),
                    torchvision.transforms.ToTensor()
                ]
            )
        )
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.preload:
            return self.images[idx].to(self.device)
        else:
            return self.images[idx][0].to(self.device)