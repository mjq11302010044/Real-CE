import torch
from torch import nn as nn
from torch.nn import functional as F
import copy
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle
import numpy as np
import cv2
import os

from torchvision.transforms import Compose, ToTensor, Normalize, Resize, InterpolationMode

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from .adain import adaptive_instance_normalization

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


@ARCH_REGISTRY.register()
class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        self.num_in_ch = num_in_ch
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # self.num_in_ch = num_in_ch
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, label_mask=None):

        if x.shape[1] > self.num_in_ch:
            x = x[:, :self.num_in_ch]

        # print("larger:", x.shape, x.shape[1], self.num_in_ch)

        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class LatentUp(nn.Module):

    def __init__(self, size_upto=8, num_in_ch=512, num_out_ch=64):
        super(LatentUp, self).__init__()

        self.upconv1 = nn.ConvTranspose2d(num_in_ch, 256, 3, stride=2, padding=1)
        self.conv1_1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.upconv4 = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.upconv5 = nn.ConvTranspose2d(64, num_out_ch, 3, stride=2, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x, image, label_mask=None):

        x = x1 = self.upconv1(x)
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        # print("x + x1:", x.shape, x1.shape)
        x = x2 = self.upconv2(self.relu(self.bn1(x + x1)))
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = x3 = self.upconv3(self.relu(self.bn2(x + x2)))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = x4 = self.upconv4(self.relu(self.bn3(x + x3)))
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.upconv5(self.relu(self.bn4(x + x4)))

        H, W = image.shape[2:]

        return F.interpolate(x, (H, W), mode="bicubic")

@ARCH_REGISTRY.register()
class CLIP_RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(CLIP_RRDBNet, self).__init__()
        self.scale = scale
        self.num_in_ch = num_in_ch
        self.clip_out_chan = 512

        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # self.num_in_ch = num_in_ch
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.model, preprocess = load_from_name("ViT-B-16", download_root='./')
        self.model.eval()

        self.latent_up = LatentUp(8, 512, num_feat)
        self.normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def padding_and_resize(self, x, o_size=256, t_size=224):

        H, W = x.shape[2:]
        pad_size = int((o_size - H)/2)

        pad_x = F.pad(x, (0, 0, pad_size, pad_size))

        return F.interpolate(pad_x, (t_size, t_size), mode="bicubic") + 1e-8

    def correlate(self, input, style):
        N, C_i, H_i, W_i = input.shape
        N, C_s, H_s, W_s = style.shape

        style = style.view(N, C_s, 1, H_s, W_s).repeat(1, 1, C_i, 1, 1)
        style = style.view(N * C_s, C_i, H_s, W_s)
        input = input.view(1, N * C_i, H_i, W_i)

        corr_feat = F.conv2d(input, style, padding="same", groups=N)

        return corr_feat.view(N, C_i, H_i, W_i)

    def forward(self, x, label_mask=None):

        if x.shape[1] > self.num_in_ch:
            x = x[:, :self.num_in_ch]

        # print("larger:", x.shape, x.shape[1], self.num_in_ch)
        with torch.no_grad():
            pad_feat = self.padding_and_resize(x)
            # pad_feat_np = pad_feat.data.cpu().numpy()
            # print("pad_feat_np:", np.unique(pad_feat_np))
            pad_feat = self.normalize(pad_feat)
            clip_feature = self.model.encode_image(pad_feat).to(torch.float32).detach()
            # clip_feature /= clip_feature.norm(dim=-1, keepdim=True)
            # clip_feature = clip_feature.data.cpu().numpy()
            # print("clip_feature:", np.unique(clip_feature))
            # clip_feature = torch.tensor(clip_feature).to(x.device)
            # clip_feature = torch.ones_like(clip_feature).float()
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        latent_weights = self.latent_up(clip_feature.unsqueeze(-1).unsqueeze(-1), feat)

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(adaptive_instance_normalization(feat, latent_weights))) #
        feat = feat + body_feat#  + self.correlate(feat, body_feat) / float(feat.shape[1])
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        if self.training:
            return out, None, None
        else:
            return out



@ARCH_REGISTRY.register()
class RRDBNetwLabelMask(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, label_mask=False):
        super(RRDBNetwLabelMask, self).__init__()
        self.upscale = scale
        self.num_in_ch = num_in_ch
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.label_mask = label_mask

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv_mask = nn.Conv2d(self.upscale ** 2, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        # self.num_in_ch = num_in_ch
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.iter = 0

    def forward(self, x, label_mask=None):

        self.iter += 1

        if x.shape[1] > self.num_in_ch:
            x = x[:, :self.num_in_ch]
        label_mask[label_mask > 0] = 1
        label_mask_np = label_mask.data.cpu().numpy()
        # print("label_mask_np:", np.unique(label_mask_np))

        largest = np.unique(label_mask_np)[-1]

        # if self.label_mask:
        N_g, C_g, H_g, W_g = label_mask.shape
        global_canvas_stack = label_mask.reshape(N_g, C_g, H_g // self.upscale, self.upscale,
                                                    W_g // self.upscale,
                                                    self.upscale)
        global_canvas_stack = global_canvas_stack.permute(0, 1, 3, 5, 2, 4)
        global_canvas_stack = global_canvas_stack.reshape(N_g, C_g * self.upscale ** 2, H_g // self.upscale,
                                                          W_g // self.upscale)

        if self.upscale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.upscale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        feat_label = self.conv_mask(global_canvas_stack)
        # print("feat:", feat.shape, feat_label.shape)
        body_feat = self.conv_body(self.body(feat + feat_label))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))

        if self.iter % 1000 == 0:

            save_im_dir = "wLabelMask_dir"
            if not os.path.isdir(save_im_dir):
                os.makedirs(save_im_dir)

            first_im = x[0, :3].permute(1, 2, 0).data.cpu().numpy()
            first_labelmask = label_mask_np[0, 0, ...]
            first_out = out[0, :3].permute(1, 2, 0).data.cpu().numpy()

            # print("first_im:", np.unique(first_im))

            cv2.imwrite(os.path.join(save_im_dir, "im_" + str(self.iter) + ".png"), (first_im * 255.0).astype(np.uint8))
            cv2.imwrite(os.path.join(save_im_dir, "imout_" + str(self.iter) + ".png"), (first_out * 255.0).astype(np.uint8))
            cv2.imwrite(os.path.join(save_im_dir, "labelmask_" + str(self.iter) + ".png"), (first_labelmask[..., None] * 255).astype(np.uint8))

        return out



class TP_RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32, tp_chan_in=64):
        super(TP_RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        self.cat_conv_1 = nn.Conv2d(tp_chan_in + num_feat, num_feat, 3, 1, 1)

    def forward(self, x, mask):

        x = torch.cat([x, mask], dim=1)
        x = self.cat_conv_1(x)
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class InfoGen(nn.Module):
    def __init__(
                self,
                t_emb,
                output_size
                 ):
        super(InfoGen, self).__init__()

        self.tconv1 = nn.ConvTranspose2d(t_emb, 512, 3, 2, bias=False)
        self.bn1 = nn.BatchNorm2d(512)

        self.tconv2 = nn.ConvTranspose2d(512, 128, 3, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(128)

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 64, 3, (2, 1), padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.tconv5 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(output_size)

    def forward(self, t_embedding):

        # t_embedding += noise.to(t_embedding.device)

        x = F.relu(self.bn1(self.tconv1(t_embedding)))
        # print(x.shape)
        x = F.relu(self.bn2(self.tconv2(x)))
        # print(x.shape)
        x = F.relu(self.bn3(self.tconv3(x)))
        # print(x.shape)
        x = F.relu(self.bn4(self.tconv4(x)))
        # print(x.shape)
        x = F.relu(self.bn5(self.tconv5(x)))

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)


class TPBranch(nn.Module):

    def __init__(self,
                 in_dim, out_dim,
                 **kwargs):
        super(TPBranch, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_dim,
                            out_channels=64,
                            kernel_size=3,
                            stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(in_channels=64,
                                out_channels=out_dim,
                                kernel_size=3,
                                stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_dim)

    def forward(self, prior):

        x = F.relu(self.bn1(self.conv_1(prior)))
        x = F.relu(self.bn2(self.conv_2(x)))
        x = F.relu(self.bn3(self.conv_3(x)))

        return x


@ARCH_REGISTRY.register()
class TP_RRDBNetV2_(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(TP_RRDBNetV2_, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.tp_chan_in = 7936
        self.tp_chan_out = 64
        self.tp_processor = InfoGen(self.tp_chan_in, self.tp_chan_out)

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body1 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body2 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body3 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body4 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body5 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body6 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body7 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body8 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        # make_layer(TP_RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch, tp_chan_in=self.tp_chan_out)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def make_binary(self, label_map):

        # label_map: [H, W], C is set to 1
        B, C, H, W = label_map.shape

        label_map[label_map > 0] = 1.0

        return label_map.float()

    def forward(self, x, label_mask):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        prior = self.tp_processor(binary_mask)
        # print("prior:", prior.shape)
        internal_feat, prior = self.body([feat, prior])
        body_feat = self.conv_body(internal_feat)
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out
