import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from . import layers
from .utils.parametrization import spectral_norm_conv

class DCGenerator(nn.Module):
    """
    Args:
        Width = 128
        Height = 64
        Channel = 1 (binary fluid/solid)
        
        dense_layers: The widths of the hidden layers of the MLP connecting 
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Control Points: `(N, 2, H_out)` where H_out = n_control_points.
            - Weights: `(N, 1, H_out)` where H_out = n_control_points.
    """
    def __init__(
        self, in_features: int, out_width: int, out_height: int,
        channels: list = [8, 4, 2, 1],
        dense_layers: list = [1024,]
        ):
        super().__init__()
        self.in_features = in_features
        self.out_width = out_width
        self.out_height = out_height
        self.channels = channels
        self.m_width, self.m_height = self.calc_m_size(out_width, out_height, channels)

        self.dense = MLP(
            in_features,
            self.m_width * self.m_height * channels[0],
            dense_layers
            )
        self.deconv = self._build_deconv()
    
    def forward(self, input):
        x = self.deconv(self.dense(input).view(-1, self.channels[0], self.m_height, self.m_width))
        return x

    def calc_m_size(self, w, h, channels):
        n_l = len(channels) - 1 
        h_out = h // (2 ** n_l)
        w_out = w // (2 ** n_l)
        return w_out, h_out

    def _build_deconv(self):
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(self.layer_sizes[:-1]):
            deconv.add_module(
                str(idx), layers.Deconv2DCombo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
        deconv.add_module(
            str(idx+1), nn.ConvTranspose2d( # type:ignore
                *self.layer_sizes[-1], 
                kernel_size=4, stride=2, padding=1
                )
            )
        return deconv
    
    @property
    def layer_sizes(self):
        return list(zip(self.channels[:-1], self.channels[1:]))

class SNDCGenerator(DCGenerator):
    def __init__(
        self, in_features: int, out_width: int, out_height: int,
        channels: list = [8, 4, 2, 1],
        dense_layers: list = [1024,]
        ):
        super().__init__(in_features, out_width, out_height, channels, dense_layers)
        self.dense = SNMLP(
            in_features,
            self.m_width * self.m_height * channels[0],
            dense_layers
            )

    def _build_deconv(self):
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(self.layer_sizes[:-1]):
            deconv.add_module(
                str(idx), layers.SNDeconv2DCombo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
        deconv.add_module(
            str(idx+1), spectral_norm(nn.ConvTranspose2d( # type:ignore
                *self.layer_sizes[-1], 
                kernel_size=4, stride=2, padding=1
                ))
            )
        return deconv
    
class TrueSNDCGenerator(DCGenerator):
    def __init__(
        self, in_features: int, out_width: int, out_height: int,
        channels: list = [8, 4, 2, 1],
        dense_layers: list = [1024,]
        ):
        super().__init__(in_features, out_width, out_height, channels, dense_layers)
        self.dense = SNMLP(
            in_features,
            self.m_width * self.m_height * channels[0],
            dense_layers
            )

    def _build_deconv(self):
        in_shapes = [(self.m_height*2**n, self.m_width*2**n) for n in range(len(self.channels) - 1)]
        deconv = nn.Sequential()
        for idx, ((in_chnl, out_chnl), in_shape) in enumerate(zip(self.layer_sizes[:-1], in_shapes[:-1])):
            deconv.add_module(
                str(idx), layers.TrueSNDeconv2DCombo(
                    in_shape, in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
        deconv.add_module(
            str(idx+1), spectral_norm_conv(nn.ConvTranspose2d( # type:ignore
                *self.layer_sizes[-1], 
                kernel_size=4, stride=2, padding=1
                ), in_shapes[-1])
            )
        return deconv

class TrueSNDCGeneratorSig(TrueSNDCGenerator):
    def forward(self, input):
        x = super().forward(input)
        return F.sigmoid(x)

class DCGeneratorSig(DCGenerator):
    def forward(self, input):
        x = super().forward(input)
        return F.sigmoid(x)

class Conv2DNetwork(nn.Module):
    """The 2D convolutional front end.
    """
    def __init__(
        self, in_channels: int, in_width: int, in_height:int, conv_channels: list, 
        combo = layers.Conv2DCombo
        ):
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_width * in_height
        self.m_features = self._calculate_m_features(conv_channels)
        self.conv = self._build_conv(conv_channels, combo)

    def forward(self, input):
        return self.conv(input)
    
    def _calculate_m_features(self, channels):
        n_l = len(channels)
        m_features = self.in_features // ((2 ** n_l)**2) * channels[-1]
        return m_features

    def _build_conv(self, channels, combo):
        conv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(
            [self.in_channels] + channels[:-1], channels
            )):
            conv.add_module(
                str(idx), combo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
            conv.add_module(str(idx+1), nn.Flatten())
        return conv

class DCDiscriminator(Conv2DNetwork):
    def __init__(
        self, in_channels: int, in_width: int, in_height:int, n_critics: int, 
        conv_channels: list, crt_layers: list
        ):
        super().__init__(in_channels, in_width, in_height, conv_channels=conv_channels)
        self.n_critics = n_critics
        self.critics = MLP(self.m_features, n_critics, crt_layers) if crt_layers is not None else nn.Identity()

    def forward(self, input):
        x = super().forward(input)
        critics = self.critics(x)
        return critics


class MLP(nn.Module):
    """Regular fully connected network generating features.
    
    Args:
        in_features: The number of input features.
        out_feature: The number of output features.
        layer_width: The widths of the hidden layers.
        combo: The layer combination to be stacked up.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output: `(N, H_out)` where H_out = out_features.
    """
    def __init__(
        self, in_features: int, out_features:int, layer_width: list, 
        combo = layers.LinearCombo
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.layer_width = list(layer_width)
        self.model = self._build_model(combo)
    
    def forward(self, input):
        return self.model(input)

    def _build_model(self, combo):
        model = nn.Sequential()
        idx = 0
        for in_ftr, out_ftr in self.layer_sizes[:-1]:
            model.add_module(str(idx), combo(in_ftr, out_ftr))
            idx += 1
        model.add_module(str(idx), nn.Linear(*self.layer_sizes[-1]))
        return model
    
    @property
    def layer_sizes(self):
        return list(zip([self.in_features] + self.layer_width, 
        self.layer_width + [self.out_features]))

class SNMLP(MLP):
    def __init__(self, in_features: int, out_features: int, layer_width: list, combo=layers.SNLinearCombo):
        super().__init__(in_features, out_features, layer_width, combo) # type: ignore
    
    def _build_model(self, combo):
        model = nn.Sequential()
        idx = 0
        for in_ftr, out_ftr in self.layer_sizes[:-1]:
            model.add_module(str(idx), combo(in_ftr, out_ftr))
            idx += 1
        model.add_module(str(idx), spectral_norm(nn.Linear(*self.layer_sizes[-1]))) # type:ignore
        return model
        