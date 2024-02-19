import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from .utils.parametrization import spectral_norm_conv

    
class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def forward(self, input):
        return self.model(input)
        
class LinearCombo(_Combo):
    r"""Regular fully connected layer combo.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            # nn.BatchNorm1d(out_features),
            # nn.ELU()
            # nn.Softplus()
            nn.LeakyReLU(alpha)
        )

class SNLinearCombo(_Combo):
    r"""Regular fully connected layer combo.
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features)),
            # nn.BatchNorm1d(out_features),
            # nn.ELU()
            # nn.Softplus()
            nn.LeakyReLU(alpha)
        )
        
class Deconv1DCombo(_Combo):
    r"""Regular deconvolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha)
        )

class Deconv2DCombo(_Combo):
    r"""Regular deconvolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

class SNDeconv2DCombo(_Combo):
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

class TrueSNDeconv2DCombo(_Combo):
    def __init__(
        self, in_shape, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2
        ):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm_conv(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding), in_shape),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha)
        )

class Conv1DCombo(_Combo):
    r"""Regular convolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2, dropout=0.4
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout)
        )

class Conv2DCombo(_Combo):
    r"""Regular convolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2, dropout=0.4
        ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha),
            # nn.Dropout(dropout)
        )

class SNConv2DCombo(_Combo):
    r"""Regular convolutional layer combo.
    """
    def __init__(
        self, in_channels, out_channels, kernel_size, 
        stride=1, padding=0, alpha=0.2, dropout=0.4
        ):
        super().__init__()
        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(alpha),
            # nn.Dropout(dropout)
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.blocks = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.activate = nn.Softplus() #nn.LeakyReLU(alpha)
    
    def forward(self, x):
        residual = x if self.should_apply_shortcut else 0
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_features == self.out_features

class SNResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.blocks = nn.Sequential(
            spectral_norm(nn.Linear(in_features, out_features))
        )
        self.activate = nn.Softplus() #nn.LeakyReLU(alpha)
    
    def forward(self, x):
        residual = x if self.should_apply_shortcut else 0
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_features == self.out_features
