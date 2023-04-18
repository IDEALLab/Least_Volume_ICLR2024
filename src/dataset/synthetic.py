import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tt
from torch.utils.data import Dataset

class ImageToyDataset(Dataset):
    def __init__(self, img_dir: str, N, 
                 degrees=90, 
                 translate=(0.2, 0.2), 
                 scale=(0.1, 0.4), 
                 shear=30, 
                 saturation=(0.5, 1), 
                 hue=0.5) -> None:
        super().__init__()
        self.image = torchvision.io.read_image(img_dir)
        self.transform(N, degrees, translate, scale, shear, saturation, hue)
    
    def transform(self, N, degrees, translate, scale, shear, saturation, hue):
        tf = nn.Sequential(
            tt.RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear),
            tt.ColorJitter(saturation=saturation, hue=hue)
        )
        imgs = [tf(self.image) for _ in range(N)]
        self.tensor = torch.stack(imgs)

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        return self.tensor[idx]