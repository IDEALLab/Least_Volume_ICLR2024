import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR10

def generate_synthetic_dataset(
            img_dir: str, N: int, 
            degrees=90, 
            translate=(0.2, 0.2), 
            scale=(0.1, 0.4), 
            shear=30, 
            saturation=(0.5, 1), 
            hue=0.5):
        image = torchvision.io.read_image(img_dir)
        tf = nn.Sequential(
            tt.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear),
            tt.ColorJitter(saturation=saturation, hue=hue)
        )
        imgs = [tf(image) for _ in range(N)]
        return torch.stack(imgs)


class ImageToyDataset(Dataset):
    def __init__(self, npy_dir: str, size=(512, 512), device='cpu') -> None: # h x w
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
    def __init__(self, device='cpu') -> None: # h x w
        super().__init__()
        self.images = MNIST(
            '../data/mnist/', train=True, download=True, 
            transform=torchvision.transforms.ToTensor()
            ) # normalized to [0, 1]
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx][0].to(self.device)
    
    
class CIFAR10Images(Dataset):
    def __init__(self, device='cpu') -> None: # h x w
        super().__init__()
        self.images = CIFAR10(
            '../data/cifar10/', train=True, download=True, 
            transform=torchvision.transforms.ToTensor()
            )
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx][0].to(self.device)