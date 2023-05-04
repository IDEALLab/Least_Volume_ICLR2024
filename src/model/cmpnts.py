import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm
from . import layers
from .flow import _RealNVP
from .utils import first_element
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

class DCGeneratorSig(nn.Module):
    def forward(self, input):
        x = super().forward(input)
        return F.sigmoid(x)

class SADCGenerator(DCGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _build_deconv(self, channels):
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(self.layer_sizes[:-1]):
            deconv.add_module(
                'conv2d_'+str(idx), layers.Deconv2DCombo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
            if out_chnl >= 8:
                deconv.add_module('attention_'+str(idx), layers.SelfAttention(out_chnl))
        deconv.add_module(
            str(idx+1), nn.ConvTranspose2d( # type:ignore
                *self.layer_sizes[-1], 
                kernel_size=4, stride=2, padding=1
                )
            )
        return deconv

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
        
class SIREN(MLP):
    def __init__(
        self, in_features: int, out_features:int, layer_width: list, omega = 30
        ):
        super().__init__(in_features, out_features, layer_width, layers.SIRENCombo) # type: ignore
        self.model[0].reset_parameters(omega) # type: ignore

class FiLMSIREN(MLP):
    def __init__(
        self, in_features: int, out_features:int, w_dim:int, layer_width: list, omega = 30
        ):
        nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
        self.w_dim = w_dim
        self.layer_width = list(layer_width)
        self.model = self._build_model(layers.FiLMSIRENBlock)
        self.model[0].reset_parameters(omega) # type:ignore
    
    def _build_model(self, combo):
        model = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(self.layer_sizes[:-1]):
            model.add_module(str(idx), combo(in_ftr, out_ftr, self.w_dim))
        model.add_module(str(idx+1), layers.FirstElement()) # type:ignore
        model.add_module(str(idx+2), nn.Linear(*self.layer_sizes[-1])) # type:ignore
        return model
    
    def forward(self, input, w):
        return self.model((input, w))
        
class DenseMLP(MLP):
    def __init__(
        self, in_features: int, out_features:int, layer_width: list, 
        combo = layers.DenseLinearCombo
        ):
        super().__init__(in_features, out_features, layer_width, combo) # type:ignore
        
    @property
    def layer_sizes(self):
        in_widths = np.cumsum([self.in_features] + self.layer_width)
        out_widths = self.layer_width + [self.out_features]
        return list(zip(in_widths, out_widths))
    
    def _build_model(self, combo):
        model = nn.Sequential()
        for idx, (in_ftr, out_ftr) in enumerate(self.layer_sizes[:-1]):
            model.add_module(str(idx), combo(in_ftr, out_ftr))
        model.add_module(str(idx+1), layers.DenseLinear(*self.layer_sizes[-1])) # type:ignore
        return model

    def forward(self, input):
        return self.model(input)[0]
        

################## Bezier #####################

class Conv1DNetwork(nn.Module):
    """The 1D convolutional front end.

    Args:
        in_channels: The number of channels of each input feature.
        in_features: The number of input features.
        conv_channels: The number of channels of each conv1d layer.

    Shape:
        - Input: `(N, C, H_in)` where C = in_channel and H_in = in_features.
        - Output: `(N, H_out)` where H_out is calculated based on in_features.
    """
    def __init__(
        self, in_channels: int, in_features: int, conv_channels: list, 
        combo = layers.Conv1DCombo
        ):
        super().__init__()
        self.in_channels = in_channels
        self.in_features = in_features
        self.m_features = self._calculate_m_features(conv_channels)
        self.conv = self._build_conv(conv_channels, combo)

    def forward(self, input):
        return self.conv(input)
    
    def _calculate_m_features(self, channels):
        n_l = len(channels)
        m_features = self.in_features // (2 ** n_l) * channels[-1]
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

class CPWGenerator(nn.Module):
    """Generate given number of control points and weights for Bezier Layer.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output. 
            Should be even.
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
        self, in_features: int, n_control_points: int,
        dense_layers: list = [1024,],
        deconv_channels: list = [96*8, 96*4, 96*2, 96],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points

        self.in_chnl, self.in_width = self._calculate_parameters(n_control_points, deconv_channels)

        self.dense = MLP(in_features, self.in_chnl*self.in_width, dense_layers)
        self.deconv = self._build_deconv(deconv_channels)
        self.cp_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 2, 1), nn.Tanh())
        self.w_gen = nn.Sequential(nn.Conv1d(deconv_channels[-1], 1, 1), nn.Sigmoid())
    
    def forward(self, input):
        x = self.deconv(self.dense(input).view(-1, self.in_chnl, self.in_width))
        cp = self.cp_gen(x)
        w = self.w_gen(x)
        return cp, w
    
    def _calculate_parameters(self, n_control_points, channels):
        n_l = len(channels) - 1
        in_chnl = channels[0]
        in_width = n_control_points // (2 ** n_l)
        assert in_width >= 4, 'Too many deconvolutional layers ({}) for the {} control points.'\
            .format(n_l, self.n_control_points)
        return in_chnl, in_width
    
    def _build_deconv(self, channels):
        deconv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(channels[:-1], channels[1:])):
            deconv.add_module(
                str(idx), layers.Deconv1DCombo(
                    in_chnl, out_chnl, 
                    kernel_size=4, stride=2, padding=1
                    )
                )
        return deconv

class BezierGenerator(nn.Module):
    """Generator for BezierGAN alike projects.

    Args:
        in_features: The number of input features.
        n_control_points: The number of control point and weights to be output.
        n_data_points: The number of data points to output.
        m_features: The number of intermediate features for generating intervals.
        feature_gen_layer: The widths of hidden layers for generating intermediate features.
        dense_layers: The widths of the hidden layers of the MLP connecting 
            input features and deconvolutional layers.
        deconv_channels: The number of channels deconvolutional layers have.
    
    Shape:
        - Input: `(N, H_in)` where H_in = in_features.
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Control Points: `(N, 2, CP)` where CP = n_control_points.
            - Weights: `(N, 1, CP)` where CP = n_control_points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """
    def __init__(
        self, in_features: int, n_control_points: int, n_data_points: int, n_curves:int = 1,
        m_features: int = 256,
        feature_gen_layers: list = [1024,],
        dense_layers: list = [1024,],
        deconv_channels: list = [96*8, 96*4, 96*2, 96],
        ):
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points

        self.feature_generator = MLP(in_features, m_features, feature_gen_layers)
        self.cpw_generators = nn.ModuleList([
            CPWGenerator(in_features, n_control_points, dense_layers, deconv_channels) for _ in range(n_curves)])
        self.bezier_layers = nn.ModuleList([
            layers.BezierLayer(m_features, n_control_points, n_data_points) for _ in range(n_curves)])
    
    def forward(self, input, verbose=False):
        features = self.feature_generator(input)
        cpws = [cpw_gen(input) for cpw_gen in self.cpw_generators]
        dpis = [bz_gen(features, cp, w) for bz_gen, (cp, w) in zip(self.bezier_layers, cpws)]
        cps, ws = zip(*cpws)
        dps, pvs, intvlls = zip(*dpis)
        if verbose:
            return torch.stack(dps), torch.stack(cps), torch.stack(ws), torch.stack(pvs), torch.stack(intvlls)
        else:
            return torch.stack(dps)
    
    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )
        

class LineGenerator(nn.Module):
    def __init__(self, N, basis, nets, nett, mask):
        super().__init__()
        self.basis = basis
        self.flows = nn.ModuleList([_RealNVP(nets, nett, mask) for i in range(N)])
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.mean, self.std = 2, 4 
        self.mean, self.std = nn.Parameter(torch.tensor(0.)), nn.Parameter(torch.tensor(1.))
    
    def normalize(self, input):
        return (input - self.mean) / self.std
        
    def forward(self, input):
        components = [self.basis(flow(input)) for flow in self.flows]
        y = torch.stack(components, dim=-1)
        weight = self.softmax(y)
        return self.normalize((y * weight).sum(-1))

class Conv2DFeature(nn.Module):
    """The input size preserving 2D convolutional module.
    """
    def __init__(
        self, conv_channels: list, 
        kernels: int=4, 
        combo = layers.Conv2DCombo
        ):
        super().__init__()
        self.kernels = kernels
        self.conv = self._build_conv(conv_channels, combo)

    def forward(self, input):
        return self.conv(input)

    def _build_conv(self, channels, combo):
        conv = nn.Sequential()
        for idx, (in_chnl, out_chnl) in enumerate(zip(
            channels[:-1], channels[1:]
            )):
            conv.add_module(
                str(idx), combo(
                    in_chnl, out_chnl, 
                    kernel_size=self.kernels, stride=1, padding='same'
                    )
                )
        return conv