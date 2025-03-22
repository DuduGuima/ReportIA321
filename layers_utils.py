import torch
import torch.nn as nn
import torch.fft
from torch.utils.checkpoint import checkpoint
import math

from torch_harmonics import *


class ComplexCardioid(nn.Module):
    """
    Complex Cardioid activation function
    """
    def __init__(self):
        super(ComplexCardioid, self).__init__()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        out = 0.5 * (1. + torch.cos(z.angle())) * z
        return out

class ComplexReLU(nn.Module):
    """
    Complex-valued variants of the ReLU activation function
    """
    def __init__(self, negative_slope=0., mode="real", bias_shape=None, scale=1.):
        super(ComplexReLU, self).__init__()
        
        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(scale * torch.ones(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope = negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:

        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)

        elif self.mode == "cardioid":
            out = 0.5 * (1. + torch.cos(z.angle())) * z

        # elif self.mode == "halfplane":
        #     # bias is an angle parameter in this case
        #     modified_angle = torch.angle(z) - self.bias
        #     condition = torch.logical_and( (0. <= modified_angle), (modified_angle < torch.pi/2.) )
        #     out = torch.where(condition, z, self.negative_slope * z)

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)
        else:
            raise NotImplementedError
            
        return out



@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class MLP(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features = None,
                 out_features = None,
                 act_layer = nn.ReLU,
                 output_bias = False,
                 drop_rate = 0.,
                 checkpointing = False,
                 gain = 1.0):
        super(MLP, self).__init__()
        self.checkpointing = checkpointing
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Fist dense layer
        fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=True)
        # initialize the weights correctly
        scale = math.sqrt(2.0 / in_features)
        nn.init.normal_(fc1.weight, mean=0., std=scale)
        if fc1.bias is not None:
            nn.init.constant_(fc1.bias, 0.0)

        # activation
        act = act_layer()

        # output layer
        fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=output_bias)
        # gain factor for the output determines the scaling of the output init
        scale = math.sqrt(gain / hidden_features)
        nn.init.normal_(fc2.weight, mean=0., std=scale)
        if fc2.bias is not None:
            nn.init.constant_(fc2.bias, 0.0)

        if drop_rate > 0.:
            drop = nn.Dropout2d(drop_rate)
            self.fwd = nn.Sequential(fc1, act, drop, fc2, drop)
        else:
            self.fwd = nn.Sequential(fc1, act, fc2)

    @torch.jit.ignore
    def checkpoint_forward(self, x):
        return checkpoint(self.fwd, x)

    def forward(self, x):
        if self.checkpointing:
            return self.checkpoint_forward(x)
        else:
            return self.fwd(x)



class SpectralConvS2(nn.Module):
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on the two-sphere S2
    using the Spherical Harmonic Transforms in torch-harmonics, but supports convolutions on the periodic
    domain via the RealFFT2 and InverseRealFFT2 wrappers.
    """

    def __init__(self,
                 forward_transform,
                 inverse_transform,
                 in_channels,
                 out_channels,
                 gain = 2.,
                 operator_type = "driscoll-healy",
                 lr_scale_exponent = 0,
                 bias = False):
        super().__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (self.forward_transform.nlat != self.inverse_transform.nlat) \
                        or (self.forward_transform.nlon != self.inverse_transform.nlon)

        # remember factorization details
        self.operator_type = operator_type

        assert self.inverse_transform.lmax == self.modes_lat
        assert self.inverse_transform.mmax == self.modes_lon

        weight_shape = [out_channels, in_channels]

        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat, self.modes_lon]
            self.contract_func = "...ilm,oilm->...olm"
        elif self.operator_type == "block-diagonal":
            weight_shape += [self.modes_lat, self.modes_lon, self.modes_lon]
            self.contract_func = "...ilm,oilnm->...oln"
        elif self.operator_type == "driscoll-healy":
            weight_shape += [self.modes_lat]
            self.contract_func = "...ilm,oil->...olm"
        else:
            raise NotImplementedError(f"Unkonw operator type f{self.operator_type}")

        # form weight tensors
        scale = math.sqrt(gain / in_channels)
        self.weight = nn.Parameter(scale * torch.randn(*weight_shape, dtype=torch.complex64))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))


    def forward(self, x):

        dtype = x.dtype
        x = x.float()
        residual = x

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                residual = self.inverse_transform(x)

        x = torch.einsum(self.contract_func, x, self.weight)

        with torch.autocast(device_type="cuda", enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias
        x = x.type(dtype)

        return x, residual