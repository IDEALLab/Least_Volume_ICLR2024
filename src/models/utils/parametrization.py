import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.parametrizations as P
from torch.nn.utils import parametrize
from typing import Union

class _SpectralNormConvNd(nn.Module):
    def __init__(self, module: nn.Module, in_shape: Union[list, tuple],     # in_shape is the dimension shape excluding batch and channel.
        n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        super().__init__()
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.eps = eps
        self.n_power_iterations = n_power_iterations
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation

        v = module.weight.new_empty((1, module.in_channels, *in_shape)).normal_(0, 1)
        u = module(v).normal_(0, 1)
        
        self.register_buffer('_u', F.normalize(u, dim=None, eps=self.eps))
        self.register_buffer('_v', F.normalize(v, dim=None, eps=self.eps))

        self._power_method(module.weight, 100)
    
    def conv(self, input):
        raise NotImplementedError
    
    def conv_T(self, input):
        raise NotImplementedError

    @torch.autograd.no_grad()
    def _power_method(self, kernel: torch.Tensor, n_power_iterations: int) -> None:
        assert kernel.ndim > 1
        for _ in range(n_power_iterations):
            self._u = F.normalize(self.conv(self._v, kernel), dim=None, eps=self.eps)   
            self._v = F.normalize(self.conv_T(self._u, kernel), dim=None, eps=self.eps) 

    def forward(self, kernel: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._power_method(kernel, self.n_power_iterations)
        u = self._u.clone(memory_format=torch.contiguous_format)
        v = self._v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u.flatten(), self.conv(v, kernel).flatten())
        return kernel / sigma

class _SpectralNormConv1d(_SpectralNormConvNd):
    def conv(self, input, kernel):
        return F.conv1d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def conv_T(self, input, kernel):
        return F.conv_transpose1d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

class _SpectralNormConv2d(_SpectralNormConvNd):
    def conv(self, input, kernel):
        return F.conv2d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def conv_T(self, input, kernel):
        return F.conv_transpose2d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

class _SpectralNormConv3d(_SpectralNormConvNd):
    def conv(self, input, kernel):
        return F.conv3d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def conv_T(self, input, kernel):
        return F.conv_transpose3d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

class _SpectralNormConvTranspose1d(_SpectralNormConvNd):
    def conv(self, input, kernel):
        return F.conv_transpose1d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def conv_T(self, input, kernel):
        return F.conv1d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

class _SpectralNormConvTranspose2d(_SpectralNormConvNd):
    def conv(self, input, kernel):
        return F.conv_transpose2d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def conv_T(self, input, kernel):
        return F.conv2d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

class _SpectralNormConvTranspose3d(_SpectralNormConvNd):
    def conv(self, input, kernel):
        return F.conv_transpose3d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def conv_T(self, input, kernel):
        return F.conv3d(input, kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)


def spectral_norm_conv(module: nn.Module, in_shape, 
        name: str = 'weight', n_power_iterations: int = 1, 
        eps: float = 1e-12, dim = None) -> nn.Module:
    kernel = getattr(module, name, None)
    if not isinstance(kernel, torch.Tensor):
        raise ValueError(
            "Module '{}' has no parameter or buffer with name '{}'".format(module, name)
        )
    if isinstance(module, nn.Conv1d):
        SNClass = _SpectralNormConv1d
    elif isinstance(module, nn.Conv2d):
        SNClass = _SpectralNormConv2d
    elif isinstance(module, nn.Conv3d):
        SNClass = _SpectralNormConv3d
    elif isinstance(module, nn.ConvTranspose1d):
        SNClass = _SpectralNormConvTranspose1d
    elif isinstance(module, nn.ConvTranspose2d):
        SNClass = _SpectralNormConvTranspose2d
    elif isinstance(module, nn.ConvTranspose3d):
        SNClass = _SpectralNormConvTranspose3d

    parametrize.register_parametrization(module, name, SNClass(module, in_shape, n_power_iterations, eps))
    return module