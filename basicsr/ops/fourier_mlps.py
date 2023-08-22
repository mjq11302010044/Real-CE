#try:
#    from models import common
#except ModuleNotFoundError:
#    import models.common as common

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
import torchpwl
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pdb

def make_model(args, parent=False):
    return CVIR(args)

class LayerNorm(nn.Module):
    def __init__(self, channels):
        super(LayerNorm, self).__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)
    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class Mlp_sequ(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1)
        # self.fc3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
    def forward(self, x):
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.fc3(out)
        return out


class Fourier_MLP(nn.Module):
    def __init__(self, channels, heads=8, shifts=4, window_size=8):
        super(Fourier_MLP, self).__init__()    
        self.channels = channels
        self.shifts   = shifts
        self.window_size = window_size
        # self.norm = LayerNorm(channels)
        self.project_inp = nn.Conv2d(channels, channels, kernel_size=1)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.mlp1= Mlp_sequ(channels//3*2)
        self.mlp2= Mlp_sequ(channels//3*2)
        self.mlp3= Mlp_sequ(channels//3*2)

        self.down = nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b,c,h,w = x.shape
        # x = self.norm(x) # has already processed outside MLP
        if self.shifts > 0:
            x = torch.roll(x, shifts=(-self.shifts, -self.shifts), dims=(2, 3))

        x = self.project_inp(x)
        x1, x2, x3 = rearrange(
            x, 'b (sc c) h w -> sc b c h w', sc=3)

        # 1: transfer to frequency domain with 16 accuracy
        x_temp = rearrange(x1, 'b c (h dh) (w dw)-> (b h w) c dh dw', dh=16, dw=16)
        x_fft1 = torch.fft.rfft2(x_temp)
        x_fft1 = torch.view_as_real(x_fft1)
        x_fft1 = rearrange(x_fft1, 'b c h w s2 -> b (c s2) h w')
        x_fft1 = self.mlp1(x_fft1)
        x_fft1 = rearrange(x_fft1, 'b (c s2) h w -> b c h w s2', s2=2)
        x_fft1 = torch.view_as_complex(x_fft1.contiguous())
        x_back1 = torch.fft.irfft2(x_fft1)
        x_back1 = rearrange(x_back1, '(b h w) c dh dw -> b c (h dh) (w dw)', 
                dh=16, dw=16, h=h//16, w=w//16)
        
        # 2: transfer to frequency domain with 32 accuracy
        x_temp = rearrange(x2, 'b c (h dh) (w dw)-> (b h w) c dh dw', dh=32, dw=32)
        x_fft2 = torch.fft.rfft2(x_temp)
        x_fft2 = torch.view_as_real(x_fft2)
        x_fft2 = rearrange(x_fft2, 'b c h w s2 -> b (c s2) h w')
        x_fft2 = self.mlp2(x_fft2)
        x_fft2 = rearrange(x_fft2, 'b (c s2) h w -> b c h w s2', s2=2)
        x_fft2 = torch.view_as_complex(x_fft2.contiguous())
        x_back2 = torch.fft.irfft2(x_fft2)
        x_back2 = rearrange(x_back2, '(b h w) c dh dw -> b c (h dh) (w dw)', 
                dh=32, dw=32, h=h//32, w=w//32)
        
        # 2: transfer to frequency domain with 64 accuracy
        x_temp = rearrange(x3, 'b c (h dh) (w dw)-> (b h w) c dh dw', dh=64, dw=64)
        x_fft3 = torch.fft.rfft2(x_temp)
        x_fft3 = torch.view_as_real(x_fft3)
        x_fft3 = rearrange(x_fft3, 'b c h w s2 -> b (c s2) h w')
        x_fft3 = self.mlp3(x_fft3)
        x_fft3 = rearrange(x_fft3, 'b (c s2) h w -> b c h w s2', s2=2)
        x_fft3 = torch.view_as_complex(x_fft3.contiguous())
        x_back3 = torch.fft.irfft2(x_fft3)
        x_back3 = rearrange(x_back3, '(b h w) c dh dw -> b c (h dh) (w dw)', 
                dh=64, dw=64, h=h//64, w=w//64)
        
        out = torch.cat((x_back1, x_back2, x_back3), dim=1)
        ca = F.avg_pool2d(out, kernel_size=(out.size(2), out.size(3)))
        ca = self.down(ca)
        ca = self.relu(ca)
        ca = self.up(ca)
        ca = F.sigmoid(ca)
        
        out = out*ca
        out = self.project_out(out)

        if self.shifts > 0:
            out = torch.roll(out, shifts=(self.shifts, self.shifts), dims=(2, 3))
        return out


