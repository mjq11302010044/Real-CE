import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.ops.elan_block import ELAB, MeanShift
from basicsr.utils.registry import ARCH_REGISTRY
import numpy as np

def create_model(args):
    return ELAN(args)

@ARCH_REGISTRY.register()
class ELAN(nn.Module):
    def __init__(self, scale=2,
              colors_in=3,
              colors_out=3,
              window_sizes=[4, 8, 16],
              m_elan=36,
              c_elan=180,
              n_share=0,
              r_expand=2,
              rgb_range=1.0):
        super(ELAN, self).__init__()

        self.scale = scale
        self.colors_in = colors_in
        self.colors_out = colors_out
        self.window_sizes = window_sizes
        self.m_elan  = m_elan
        self.c_elan  = c_elan
        self.n_share = n_share
        self.r_expand = r_expand
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.rgb_chan = 3
        self.rgb_chan_out = 3

        # define head module
        m_head = [nn.Conv2d(self.colors_in, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1+self.n_share)):
            if (i+1) % 2 == 1: 
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 0, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:              
                m_body.append(
                    ELAB(
                        self.c_elan, self.c_elan, self.r_expand, 1, 
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors_out*self.scale*self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, mask=None):

        C, H, W = x.shape[1:]
        x = x[:, :self.colors_in]
        x = self.check_image_size(x)
        '''
        if C > self.rgb_chan:
            rgb = x[:, :3]
            canny = x[:, 3:]
            rgb = self.sub_mean(rgb)
            x = torch.cat([rgb, canny], dim=1)
        else:
            x = self.sub_mean(x)
        '''
        x = self.head(x)
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        '''
        if C > self.rgb_chan_out:
            rgb = x[:, :3]
            canny = x[:, 3:]
            rgb = self.add_mean(rgb)
            x = torch.cat([rgb, canny], dim=1)
        else:
            x = self.add_mean(x)
        '''
        # x_np = x[:, :3].data.cpu().numpy()
        # print(x.shape, np.unique(x_np))

        return x[:, :, 0:H*self.scale, 0:W*self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize*self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


if __name__ == '__main__':
    pass