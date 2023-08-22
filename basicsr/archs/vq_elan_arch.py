import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.ops.elan_block import ELAB, MeanShift, ELAB_wP
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.ops.quantize import VectorQuantizer
from .quant_swinir_arch import SwinTransformerBlock, PatchUnEmbed


def create_model(args):
    return ELAN(args)

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops

class VQ_Encoder(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 **kwargs):

        super(VQ_Encoder, self).__init__()

        self.patch_norm = patch_norm
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.quant_block1 = RSTB(
            dim=embed_dim,
            input_resolution=(patches_resolution[0] // 2, patches_resolution[1] // 2),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection)

        self.quant_block2 = RSTB(
            dim=embed_dim,
            input_resolution=(patches_resolution[0] // 4, patches_resolution[1] // 4),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection)

        self.quant_block3 = RSTB(
            dim=embed_dim * 2,
            input_resolution=(patches_resolution[0] // 8, patches_resolution[1] // 8),
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],  # no impact on SR results
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection)

        self.down_conv1 = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(embed_dim, embed_dim * 2, 3, 2, 1)

    def forward(self, x_):

        N, C_, H_ori, W_ori = x_.shape

        x = self.down_conv1(x_)
        # x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[2] * x.shape[3], x.shape[1])
        x = self.quant_block1(x, (H_ori // 2, W_ori // 2))
        x = x.reshape(N, -1, H_ori // 2, W_ori // 2)

        x = self.down_conv2(x)
        # x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[2] * x.shape[3], x.shape[1])
        x = self.quant_block2(x, (H_ori // 4, W_ori // 4))
        x = x.reshape(N, -1, H_ori // 4, W_ori // 4)

        x = self.down_conv3(x)
        # x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[2] * x.shape[3], x.shape[1])
        x = self.quant_block3(x, (H_ori // 8, W_ori // 8))
        x = x.reshape(N, -1, H_ori // 8, W_ori // 8)

        return x


class VQ_ELAB_Encoder(nn.Module):

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=96,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 window_sizes=[4, 8, 16],
                 **kwargs):

        super(VQ_ELAB_Encoder, self).__init__()

        self.patch_norm = patch_norm
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.num_layers = len(depths)
        self.embed_dim = in_chans
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        # , inp_channels, out_channelss

        self.quant_block1 = ELAB(
            inp_channels = embed_dim,
            out_channels = embed_dim,
            window_sizes=window_sizes
            )

        self.quant_block2 = ELAB(
            inp_channels=embed_dim,
            out_channels=embed_dim,
            window_sizes=window_sizes
        )

        self.quant_block3 = ELAB(
            inp_channels=embed_dim * 2,
            out_channels=embed_dim * 2,
            window_sizes=[i // 2 for i in window_sizes]

        )

        self.down_conv1 = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.down_conv2 = nn.Conv2d(embed_dim, embed_dim, 3, 2, 1)
        self.down_conv3 = nn.Conv2d(embed_dim, embed_dim * 2, 3, 2, 1)

    def forward(self, x_):

        N, C_, H_ori, W_ori = x_.shape

        x = self.down_conv1(x_)
        # x = self.patch_embed(x)
        # x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[2] * x.shape[3], x.shape[1])
        x = self.quant_block1(x)
        # x = x.reshape(N, -1, H_ori // 2, W_ori // 2)

        x = self.down_conv2(x)
        # x = self.patch_embed(x)
        # x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[2] * x.shape[3], x.shape[1])
        x = self.quant_block2(x)
        # x = x.reshape(N, -1, H_ori // 4, W_ori // 4)

        x = self.down_conv3(x)
        # x = self.patch_embed(x)
        # x = x.permute(0, 2, 3, 1).reshape(-1, x.shape[2] * x.shape[3], x.shape[1])
        x = self.quant_block3(x)
        # x = x.reshape(N, -1, H_ori // 8, W_ori // 8)

        return x

class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops

@ARCH_REGISTRY.register()
class VQ_ELAN(nn.Module):
    def __init__(self, scale=2,
                 colors=3,
                 window_sizes=[4, 8, 16],
                 m_elan=36,
                 c_elan=180,
                 n_share=0,
                 r_expand=2,
                 rgb_range=255):
        super(VQ_ELAN, self).__init__()

        self.scale = scale
        self.colors = colors
        self.window_sizes = window_sizes
        self.m_elan = m_elan
        self.c_elan = c_elan
        self.n_share = n_share
        self.r_expand = r_expand
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.quant_in_dim = 64
        self.quant_dim = 1024
        self.before_quant_conv = nn.Conv2d(c_elan, self.quant_in_dim, 1)

        img_size = 64
        embed_dim = 180
        patch_size = 1
        in_chans = c_elan
        depths = (4, 4, 4, 4, 4, 4)
        num_heads = (4, 4, 4, 4, 4, 4)
        window_size = 8
        mlp_ratio = 4
        patch_norm = True
        self.vq_enc = VQ_Encoder(img_size,
                                 patch_size,
                                 in_chans,
                                 embed_dim=self.quant_in_dim,
                                 depths=depths,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 patch_norm=patch_norm,
                                 )

        self.quantizer = VectorQuantizer(self.quant_dim, self.quant_in_dim * 2, 0.25)
        self.after_quant_conv = nn.Conv2d(self.quant_in_dim * 2, embed_dim, 1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1 + self.n_share)):
            if (i + 1) % 2 == 1:
                m_body.append(
                    ELAB_wP(
                        self.c_elan, self.c_elan, self.r_expand, 0,
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:
                m_body.append(
                    ELAB_wP(
                        self.c_elan, self.c_elan, self.r_expand, 1,
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.sub_mean(x)
        x = self.head(x)

        x_origin = x

        # Quantizer encoder
        # quant_in = self.quant_block(x)
        quant_in = self.before_quant_conv(x_origin)
        # Downsample to 1/8 spatial size 8 * 8
        quant_in = self.vq_enc(quant_in)
        quant_out, emb_loss, _ = self.quantizer(quant_in)
        quant_out = self.after_quant_conv(quant_out)

        N, C, xH, xW = x_origin.shape
        N, C, qH, qW = quant_out.shape
        x_origin = x_origin.reshape(1, N * C, xH, xW)
        quant_out = quant_out.reshape(N * C, 1, qH, qW)
        # A correlation
        quant_corr = nn.functional.conv2d(x_origin, quant_out, stride=1, padding=(qH // 2, qW // 2), groups=N * C)
        quant_corr = nn.functional.interpolate(quant_corr, (xH, xW), mode="bicubic")
        quant_corr = quant_corr.reshape(N, C, xH, xW)# \
            #.permute(0, 2, 3, 1) \
            #.reshape(N, xH * xW, C)

        res_x, res_corr = self.body([x, quant_corr])
        res = res_x + x
        x = self.tail(res)
        x = self.add_mean(x)

        if self.training:
            return x[:, :, 0:H * self.scale, 0:W * self.scale], emb_loss
        else:
            return x[:, :, 0:H * self.scale, 0:W * self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
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


@ARCH_REGISTRY.register()
class VQ_ELANv2(nn.Module):
    def __init__(self, scale=2,
                 colors=3,
                 window_sizes=[4, 8, 16],
                 m_elan=36,
                 c_elan=180,
                 n_share=0,
                 r_expand=2,
                 rgb_range=255):
        super(VQ_ELANv2, self).__init__()

        self.scale = scale
        self.colors = colors
        self.window_sizes = window_sizes
        self.m_elan = m_elan
        self.c_elan = c_elan
        self.n_share = n_share
        self.r_expand = r_expand
        self.sub_mean = MeanShift(rgb_range)
        self.add_mean = MeanShift(rgb_range, sign=1)

        self.quant_in_dim = 180
        self.quant_dim = 1024
        self.before_quant_conv = nn.Conv2d(c_elan, self.quant_in_dim, 1)

        img_size = 64
        embed_dim = 180
        patch_size = 1
        in_chans = c_elan
        depths = (4, 4, 4, 4, 4, 4)
        num_heads = (4, 4, 4, 4, 4, 4)
        window_size = 8
        mlp_ratio = 4
        patch_norm = True
        self.vq_enc = VQ_ELAB_Encoder(img_size,
                                 patch_size,
                                 in_chans,
                                 embed_dim=self.quant_in_dim,
                                 depths=depths,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 mlp_ratio=mlp_ratio,
                                 patch_norm=patch_norm,
                                 window_sizes=window_sizes
                                 )

        self.quantizer = VectorQuantizer(self.quant_dim, self.quant_in_dim * 2, 0.25)
        self.after_quant_conv = nn.Conv2d(self.quant_in_dim * 2, embed_dim, 1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_elan, kernel_size=3, stride=1, padding=1)]

        # define body module
        m_body = []
        for i in range(self.m_elan // (1 + self.n_share)):
            if (i + 1) % 2 == 1:
                m_body.append(
                    ELAB_wP(
                        self.c_elan, self.c_elan, self.r_expand, 0,
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
            else:
                m_body.append(
                    ELAB_wP(
                        self.c_elan, self.c_elan, self.r_expand, 1,
                        self.window_sizes, shared_depth=self.n_share
                    )
                )
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_elan, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.sub_mean(x)
        x = self.head(x)

        x_origin = x

        # Quantizer encoder
        # quant_in = self.quant_block(x)
        quant_in = self.before_quant_conv(x_origin)
        # Downsample to 1/8 spatial size 8 * 8
        quant_in = self.vq_enc(quant_in)
        quant_out, emb_loss, _ = self.quantizer(quant_in)
        quant_out = self.after_quant_conv(quant_out)

        N, C, xH, xW = x_origin.shape
        N, C, qH, qW = quant_out.shape
        x_origin = x_origin.reshape(1, N * C, xH, xW)
        quant_out = quant_out.reshape(N * C, 1, qH, qW)
        # A correlation
        quant_corr = nn.functional.conv2d(x_origin, quant_out, stride=1, padding=(qH // 2, qW // 2), groups=N * C)
        quant_corr = nn.functional.interpolate(quant_corr, (xH, xW), mode="bicubic")
        quant_corr = quant_corr.reshape(N, C, xH, xW)# \
            #.permute(0, 2, 3, 1) \
            #.reshape(N, xH * xW, C)

        res_x, res_corr = self.body([x, quant_corr])
        res = res_x + x
        x = self.tail(res)
        x = self.add_mean(x)

        if self.training:
            return x[:, :, 0:H * self.scale, 0:W * self.scale], emb_loss
        else:
            return x[:, :, 0:H * self.scale, 0:W * self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
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