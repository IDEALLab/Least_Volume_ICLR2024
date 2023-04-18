import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm
from math import sqrt
from .utils import first_element
from .utils.parametrization import spectral_norm_conv

_eps = 1e-7

class BezierLayer(nn.Module):
    r"""Produces the data points on the Bezier curve, together with coefficients 
        for regularization purposes.

    Args:
        in_features: size of each input sample.
        n_control_points: number of control points.
        n_data_points: number of data points to be sampled from the Bezier curve.

    Shape:
        - Input: 
            - Input Features: `(N, H)` where H = in_features.
            - Control Points: `(N, D, CP)` where D stands for the dimension of Euclidean space, 
            and CP is the number of control points. For 2D applications, D = 2.
            - Weights: `(N, 1, CP)` where CP is the number of control points. 
        - Output:
            - Data Points: `(N, D, DP)` where D is the dimension and DP is the number of data points.
            - Parameter Variables: `(N, 1, DP)` where DP is the number of data points.
            - Intervals: `(N, DP)` where DP is the number of data points.
    """

    def __init__(self, in_features: int, n_control_points: int, n_data_points: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_control_points = n_control_points
        self.n_data_points = n_data_points
        self.generate_intervals = nn.Sequential(
            nn.Linear(in_features, n_data_points-1),
            nn.Softmax(dim=1),
            nn.ConstantPad1d([1,0], 0)
        )

    def forward(self, input: Tensor, control_points: Tensor, weights: Tensor) -> Tensor:
        cp, w = self._check_consistency(control_points, weights) # [N, d, n_cp], [N, 1, n_cp]
        bs, pv, intvls = self.generate_bernstein_polynomial(input) # [N, n_cp, n_dp]
        dp = (cp * w) @ bs / (w @ bs) # [N, d, n_dp]
        return dp, pv, intvls
    
    def _check_consistency(self, control_points: Tensor, weights: Tensor) -> Tensor:
        assert control_points.shape[-1] == self.n_control_points, 'The number of control points is not consistent.'
        assert weights.shape[-1] == self.n_control_points, 'The number of weights is not consistent.'
        assert weights.shape[1] == 1, 'There should be only one weight corresponding to each control point.'
        return control_points, weights

    def generate_bernstein_polynomial(self, input: Tensor) -> Tensor:
        intvls = self.generate_intervals(input) # [N, n_dp]
        pv = torch.cumsum(intvls, -1).clamp(0, 1).unsqueeze(1) # [N, 1, n_dp]
        pw1 = torch.arange(0., self.n_control_points, device=input.device).view(1, -1, 1) # [1, n_cp, 1]
        pw2 = torch.flip(pw1, (1,)) # [1, n_cp, 1]
        lbs = pw1 * torch.log(pv+_eps) + pw2 * torch.log(1-pv+_eps) \
            + torch.lgamma(torch.tensor(self.n_control_points, device=input.device)+_eps).view(1, -1, 1) \
            - torch.lgamma(pw1+1+_eps) - torch.lgamma(pw2+1+_eps) # [N, n_cp, n_dp]
        bs = torch.exp(lbs) # [N, n_cp, n_dp]
        return bs, pv, intvls

    def extra_repr(self) -> str:
        return 'in_features={}, n_control_points={}, n_data_points={}'.format(
            self.in_features, self.n_control_points, self.n_data_points
        )

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.rlps_conv = nn.Conv2d(in_channels=2, out_channels=in_channels // 8, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x, output_attention=False):
        query = self.query_conv(x).flatten(2)
        key = self.key_conv(x).flatten(2)
        rp_code = self.relative_position(x)
        score = torch.bmm(query.transpose(1, 2), key) / sqrt(key.shape[1])
        rp_score = torch.bmm(query.transpose(1, 2).unsqueeze(-2).flatten(0, 1), \
            rp_code.transpose(1, 2).flatten(0, 1)).reshape(score.shape) / sqrt(key.shape[1])
        attention = F.softmax(score + rp_score, dim=-1)
        value = self.value_conv(x).flatten(2)
        output = self.gamma * torch.bmm(value, attention.transpose(1, 2)).view(x.shape) + x
        
        if output_attention:
            return output, attention
        else:
            return output
    
    def relative_position(self, x):
        """ Assume the input is of shape B x C x N1 x N2"""
        axes = [torch.linspace(0, 1, dim).to(x.device) for dim in x.shape[2:]]
        coords = torch.stack(torch.meshgrid(axes)).flatten(1).unsqueeze(0)
        rp_embedding = self.rlps_conv(coords.unsqueeze(-1) - coords.unsqueeze(-2))
        return rp_embedding.expand(len(x), -1, -1, -1)

class Sin(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.sin(input)

class FirstElement(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return first_element(input)
    
class _Combo(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = None
    
    def forward(self, input):
        return self.model(input)
        
class _DenseCombo(_Combo):
    def forward(self, inputs):
        if isinstance(inputs, tuple):
            return (self.model(torch.cat(inputs, dim=1)),) + inputs
        else:
            return self.model(inputs), inputs
        
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

class SIRENCombo(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()
    
    def reset_parameters(self, omega=1):
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.linear.weight)
        init.uniform_(self.linear.weight, -sqrt(6/fan_in)*omega, sqrt(6/fan_in)*omega)
    
    def forward(self, input):
        return torch.sin(self.linear(input))

class FiLMSIRENBlock(SIRENCombo):
    def __init__(self, in_features, out_features, w_dim):
        super().__init__(in_features, out_features)
        self.affine = nn.Linear(w_dim, 2 * out_features)
    
    def forward(self, input):
        x, w = input
        h = self.linear(x)
        gamma, beta = self.affine(w).split(self.linear.out_features, -1)
        return torch.sin(gamma * h + beta), w
        
class DenseLinearCombo(_DenseCombo):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(alpha)
            )

class DenseLinear(_DenseCombo):
    def __init__(self, in_features, out_features, alpha=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, out_features)
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

        
class LineLayer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float))
        self.std = nn.Parameter(torch.tensor(std, dtype=torch.float))
        self.normal = torch.distributions.Normal(self.mean, self.std)
    
    def forward(self, input):
        lp = self.normal.log_prob(input[:, :1])
        return torch.exp(lp)

class NestedDropout(nn.Module):
    def __init__(self, prior=0.1):
        super().__init__()
        self.prior = torch.distributions.Geometric(prior) if isinstance(prior, float) else prior
    
    def forward(self, input):
        if self.training:
            idx = int(self.prior.sample() + 1)
            input[:, idx:] = 0.
            return input
        else:
            return input
