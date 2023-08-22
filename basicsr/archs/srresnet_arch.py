from torch import nn as nn
from torch.nn import functional as F
import torch
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import ResidualBlockNoBN, default_init_weights, make_layer, RecurrentResidualBlockTL
from .rrdbnet_arch import RRDBNet, TP_RRDB, CLIP_RRDBNet
from .restormer_arch import Restormer
from .scunet_arch import SCUNet, SCUNet_v2
from basicsr.ops.roi_align_rotated import ROIAlignRotated
from basicsr.ops.roi_align_rotated_inverse import ROIAlignRotatedInverse
from basicsr.ops.adain import adaptive_instance_normalization as adain
import cv2,os
import numpy as np
from torchvision.transforms import Pad
from easydict import EasyDict

import basicsr.metrics.crnn as crnn

global_device="cuda:0"
opt = {
        "Transformation": 'None',
        "FeatureExtraction": 'ResNet',
        "SequenceModeling": 'None',
        "Prediction": 'CTC',
        "num_fiducial": 20,
        "input_channel": 1,
        "output_channel": 512,
        "hidden_size": 256,
        "saved_model": "/home/majianqi/workspace/BasicSR/basicsr/metrics/scene_base_CRNN.pth",#"best_accuracy.pth", #"None-ResNet-None-CTC.pth",#"CRNN-PyTorchCTC.pth", # None-ResNet-None-CTC.pth
        "saved_model_eng": "/home/majianqi/workspace/BasicSR/basicsr/metrics/crnn.pth",
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz",
        "character_eng": "-0123456789abcdefghijklmnopqrstuvwxyz"
    }

opt['character'] = open("/home/majianqi/workspace/BasicSR/basicsr/metrics/al_chinese.txt", 'r').readlines()[0].replace("\n", "")
opt["num_class"] = len(opt['character'])
opt = EasyDict(opt)

def CRNN_init(recognizer_path=None, opt=None):

    alphabet = open("/home/majianqi/workspace/BasicSR/basicsr/metrics/benchmark.txt", 'r').readlines()[0].replace("\n", "")

    model = crnn.CRNN(3, 256, len(alphabet) + 1, 32)
    model = model.to(global_device)
    # model.eval()
    # cfg = self.config.TRAIN
    # aster_info = AsterInfo(cfg.voc_type)
    model_path = recognizer_path if not recognizer_path is None else opt.saved_model
    print('loading pretrained TPG model from %s' % model_path)
    stat_dict = torch.load(model_path)
    # model.load_state_dict(stat_dict)

    load_keys = stat_dict.keys()
    man_load_dict = model.state_dict()
    for key in stat_dict:
        if not key.replace("module.", "") in man_load_dict:
            print("Key not match", key, key.replace("module.", ""))
        man_load_dict[key.replace("module.", "")] = stat_dict[key]
    model.load_state_dict(man_load_dict)


    return model

def parse_CRNN_data(imgs_input_, ratio_keep=True):

    in_width = 512

    if ratio_keep:
        real_height, real_width = imgs_input_.shape[2:]
        ratio = real_width / float(real_height)

        # if ratio > 3:
        in_width = max(min(int(ratio * 32), 1024), 16)
    imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

    return imgs_input


@ARCH_REGISTRY.register()
class MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)

        if self.upscale in [2, 3, 4]:
            default_init_weights(self.upconv1, 0.1)

        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

        self.internal = None

    def get_internal_features(self):
        return self.internal

    def forward_feature(self, x, labelmask=None):
        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        return out

    def forward(self, x, labelmask=None):

        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        self.internal = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base[:, :self.num_out_ch]
        return out


@ARCH_REGISTRY.register()
class LocalEnhancedMSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_out_ch_local=3, num_feat=64, num_block=16, upscale=4):
        super(LocalEnhancedMSRResNet, self).__init__()
        self.upscale = upscale
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_out_ch_local = num_out_ch_local

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.conv_sec = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.bn_conv_sec = nn.BatchNorm2d(num_feat)

        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.local_conv1 = nn.Conv2d(num_out_ch_local, num_feat, 3, 1, 1)
        self.bn_local_conv1 = nn.BatchNorm2d(num_feat)
        self.local_dconv1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.bn_local_dconv1 = nn.BatchNorm2d(num_feat)
        self.local_dconv2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.bn_local_dconv2 = nn.BatchNorm2d(num_feat)

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)

        if self.upscale in [2, 3, 4]:
            default_init_weights(self.upconv1, 0.1)

        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

        self.internal = None

    def get_internal_features(self):
        return self.internal

    def forward_feature(self, x, labelmask=None):
        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        return out

    def forward(self, x, x_local, labelmask=None):

        x_local = x_local[:, :self.num_out_ch]
        f1_local = self.lrelu(self.bn_local_conv1(self.local_conv1(x_local)))
        f1d_local = self.lrelu(self.bn_local_dconv1(self.local_dconv1(f1_local)))
        f2d_local = self.lrelu(self.bn_local_dconv2(self.local_dconv2(f1d_local)))

        x = x[:, :self.num_in_ch]
        feat = self.lrelu(self.conv_first(x))
        # print("feat:", feat.shape, f2d_local.shape)
        feat = self.lrelu(self.conv_sec(torch.cat([feat, f2d_local], dim=1)))
        out = self.body(feat)

        self.internal = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(torch.cat([out, f2d_local], dim=1))))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(torch.cat([out, f1d_local], dim=1))))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(torch.cat([out, f1_local], dim=1))))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base[:, :self.num_out_ch]
        return out


@ARCH_REGISTRY.register()
class HAMO_MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(HAMO_MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.num_in_ch = num_in_ch

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)

        if self.upscale in [2, 3, 4]:
            default_init_weights(self.upconv1, 0.1)

        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

        self.internal = None

    def get_internal_features(self):
        return self.internal

    def forward_feature(self, x, labelmask=None):
        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        return out

    def forward(self, x, labelmask=None):

        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)

        self.internal = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        # out += base
        return out


@ARCH_REGISTRY.register()
class TP_MSRResNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, upscale=4):
        super(TP_MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.body1 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        self.body2 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        self.body3 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        self.body4 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        self.body5 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        # self.body6 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)

        self.num_in_ch = num_in_ch

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.recognizer_student = CRNN_init(opt=opt)
        self.recognizer_teacher = CRNN_init(opt=opt)
        self.tp_transform = InfoGen(7935, 64)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)

        if self.upscale in [2, 3, 4]:
            default_init_weights(self.upconv1, 0.1)

        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

        self.internal = None

    def get_internal_features(self):
        return self.internal, self.x_rec, self.y_rec

    def forward_feature(self, x, labelmask=None):
        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        return out

    def forward(self, x, y=None, labelmask=None):

        x = x[:, :self.num_in_ch]
        self.y_rec = None
        if not y is None:
            y = y[:, :self.num_in_ch]
            parse_y = parse_CRNN_data(y)
            with torch.no_grad():
                self.y_rec = self.recognizer_teacher(parse_y).detach()
                self.y_rec = torch.nn.functional.softmax(self.y_rec, -1)
                self.y_rec = self.y_rec.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        parse_x = parse_CRNN_data(x)

        self.x_rec = self.recognizer_student(parse_x)
        # print("x_rec:", self.x_rec.shape)

        self.x_rec = torch.nn.functional.softmax(self.x_rec, -1)
        self.x_rec = self.x_rec.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

        tp_feature, pre_weights = self.tp_transform(self.x_rec)
        tp_feature = F.interpolate(tp_feature, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        #####################

        # print("tp_feature:", tp_feature.shape)

        feat = self.lrelu(self.conv_first(x))
        # out = self.body(feat) # + tp_feature

        feat = self.body1(feat, tp_feature)
        feat = self.body2(feat, tp_feature)
        feat = self.body3(feat, tp_feature)
        feat = self.body4(feat, tp_feature)
        out = self.body5(feat, tp_feature)
        # out = self.body6(feat, tp_feature)

        self.internal = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base

        if self.training:
            return out, self.x_rec, self.y_rec
        else:
            return out


@ARCH_REGISTRY.register()
class TP_RRDBNetV2(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(TP_RRDBNetV2, self).__init__()
        self.upscale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        # self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        self.body1 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body2 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body3 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body4 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body5 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body6 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body7 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)
        self.body8 = TP_RRDB(num_feat, num_grow_ch=num_grow_ch, tp_chan_in=64)

        self.num_in_ch = num_in_ch

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.recognizer_student = CRNN_init(opt=opt)
        self.recognizer_teacher = CRNN_init(opt=opt)
        self.tp_transform = InfoGen(7935, 64)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.alphabet = open("/home/majianqi/workspace/BasicSR/basicsr/metrics/benchmark.txt", 'r').readlines()[0].replace("\n", "")
        self.alphabet = "_" + self.alphabet

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)

        if self.upscale in [2, 3, 4]:
            default_init_weights(self.upconv1, 0.1)

        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

        self.internal = None

    def get_internal_features(self):
        return self.internal, self.x_rec, self.y_rec

    def get_onehot(self, probs):

        N, C, H, W = probs.shape
        probs_max, probs_argmax = probs.max(dim=1)
        onehot_prior = F.one_hot(probs_argmax, C)
        onehot_prior = onehot_prior.permute(0, 3, 1, 2)

        return onehot_prior.float()

    def forward_feature(self, x, labelmask=None):
        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        return out

    def forward(self, x, y=None, labelmask=None):

        x = x[:, :self.num_in_ch]
        self.y_rec = None
        if not y is None:
            y = y[:, :self.num_in_ch]
            parse_y = parse_CRNN_data(y)
            with torch.no_grad():
                self.y_rec = self.recognizer_teacher(parse_y).detach()
                self.y_rec = torch.nn.functional.softmax(self.y_rec, -1)
                self.y_rec = self.y_rec.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        parse_x = parse_CRNN_data(x)

        self.x_rec = self.recognizer_student(parse_x)
        # print("x_rec:", self.x_rec.shape)

        self.x_rec = torch.nn.functional.softmax(self.x_rec, -1)
        self.x_rec = self.x_rec.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)

        # if self.training and not y is None:
        self.onehot_rec = self.get_onehot(self.y_rec)
        # else:
        # self.onehot_rec = self.get_onehot(self.x_rec)

        '''
        print("x_rec", self.x_rec.shape)
        x_rec_np = self.x_rec.data.cpu().numpy()
        x_rec_argmax = np.argmax(x_rec_np, axis=1)
        x_rec_max = np.max(x_rec_np, axis=1)

        pred_labels_ = x_rec_argmax[0].reshape(-1)
        pred_probs = x_rec_max[0].reshape(-1)

        pred_labels = pred_labels_[pred_labels_ != 0]
        pred_probs = pred_probs[pred_labels_ != 0]
        print("pred_labels_x:", [self.alphabet[idx] for idx in pred_labels], pred_probs)
        
        x_rec_np = self.y_rec.data.cpu().numpy()
        x_rec_argmax = np.argmax(x_rec_np, axis=1)
        x_rec_max = np.max(x_rec_np, axis=1)

        pred_labels_ = x_rec_argmax[0].reshape(-1)
        pred_probs = x_rec_max[0].reshape(-1)

        pred_labels = pred_labels_[pred_labels_ != 0]
        pred_probs = pred_probs[pred_labels_ != 0]
        print("pred_labels_y:", [self.alphabet[idx] for idx in pred_labels], pred_probs)
        '''
        tp_feature, pre_weights = self.tp_transform(self.onehot_rec.detach())
        tp_feature = F.interpolate(tp_feature, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        #####################

        # print("tp_feature:", tp_feature.shape)

        feat = self.lrelu(self.conv_first(x))
        # out = self.body(feat) # + tp_feature

        feat = self.body1(feat, tp_feature)
        feat = self.body2(feat, tp_feature)
        feat = self.body3(feat, tp_feature)
        feat = self.body4(feat, tp_feature)
        feat = self.body5(feat, tp_feature)
        feat = self.body6(feat, tp_feature)
        feat = self.body7(feat, tp_feature)
        out = self.body8(feat, tp_feature)
        # out = self.body6(feat, tp_feature)

        self.internal = out

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base

        if self.training:
            return out, self.x_rec, self.y_rec
        else:
            return out


@ARCH_REGISTRY.register()
class TB_RRDBNet(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """
    # num_in_ch: 3
    # num_out_ch: 3
    # scale: 2
    # num_feat: 64
    # num_block: 23
    # num_grow_ch: 32
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=16, scale=4, num_grow_ch=32):
        super(TB_RRDBNet, self).__init__()
        self.upscale = scale

        #self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        #self.body = make_layer(ResidualBlockNoBN, num_block, num_feat=num_feat)

        #self.body1 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        #self.body2 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        #self.body3 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        #self.body4 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        #self.body5 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)
        # self.body6 = RecurrentResidualBlockTL(num_feat=num_feat, text_channels=num_feat)

        self.rrdbnet = RRDBNet(num_in_ch=num_in_ch, num_out_ch=num_out_ch, num_feat=num_feat, num_block=num_block, scale=scale, num_grow_ch=num_grow_ch)

        self.num_in_ch = num_in_ch

        # self.recognizer_student = CRNN_init(opt=opt)
        # self.recognizer_teacher = CRNN_init(opt=opt)
        # self.tp_transform = InfoGen(7935, 64)
        '''
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.conv_first, self.conv_hr, self.conv_last], 0.1)

        if self.upscale in [2, 3, 4]:
            default_init_weights(self.upconv1, 0.1)

        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)
        '''
        self.internal = None

    def get_internal_features(self):
        return self.internal, self.x_rec, self.y_rec

    def forward_feature(self, x, labelmask=None):
        x = x[:, :self.num_in_ch]

        feat = self.lrelu(self.conv_first(x))
        out = self.body(feat)
        return out

    def forward(self, x, y=None, labelmask=None):

        '''
        x = x[:, :self.num_in_ch]
        self.y_rec = None
        if not y is None:
            y = y[:, :self.num_in_ch]
            parse_y = parse_CRNN_data(y[:, :3])
            with torch.no_grad():
                self.y_rec = self.recognizer_teacher(parse_y).detach()
                self.y_rec = torch.nn.functional.softmax(self.y_rec, -1)
                self.y_rec = self.y_rec.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        parse_x = parse_CRNN_data(x[:, :3])

        self.x_rec = self.recognizer_student(parse_x)
        # print("x_rec:", self.x_rec.shape)

        self.x_rec = torch.nn.functional.softmax(self.x_rec, -1)
        self.x_rec = self.x_rec.permute(1, 0, 2).unsqueeze(1).permute(0, 3, 1, 2)
        '''
        # tp_feature, pre_weights = self.tp_transform(self.x_rec)
        # tp_feature = F.interpolate(tp_feature, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=True)

        #####################

        #feat = self.lrelu(self.conv_first(x))
        #out = self.body(feat) # + tp_feature

        out = self.rrdbnet(x)

        self.internal = out

        if self.training:
            return out, torch.tensor(0.0).to(x.device), torch.tensor(0.0).to(x.device) # self.x_rec, self.y_rec
        else:
            return out



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

        self.tconv3 = nn.ConvTranspose2d(128, 64, 3, (2, 1), padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        self.tconv4 = nn.ConvTranspose2d(64, 64, 3, (2, 1), padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        #self.tconv5 = nn.ConvTranspose2d(64, output_size, 3, (2, 1), padding=0, bias=False)
        #self.bn5 = nn.BatchNorm2d(output_size)

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
        #x = F.relu(self.bn5(self.tconv5(x)))

        return x, torch.zeros((x.shape[0], 1024, t_embedding.shape[-1])).to(x.device)


@ARCH_REGISTRY.register()
class TE_STISR(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_in_ch_local=3, num_out_ch_local=3, num_feat=64, num_block=16, upscale=4, cropped_upscale=2, out_scale=(256, 256)):
        super(TE_STISR, self).__init__()
        self.upscale = upscale
        self.rroi_scale = (32, 128) #
        self.inverse_rroi_scale = (out_scale[0]//upscale, out_scale[1]//upscale)

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.cropped_upscale = cropped_upscale

        self.text_recovery_net = TB_RRDBNet(num_in_ch=num_in_ch_local,
                                            num_out_ch=num_out_ch_local,
                                            scale=cropped_upscale,
                                            num_feat=num_feat,
                                            num_block=23,
                                            num_grow_ch=32) # TP_MSRResNet(num_block=8, upscale=cropped_upscale, num_feat=64)
        self.global_sr_net = MSRResNet(num_in_ch=num_in_ch, num_out_ch=num_out_ch, upscale=upscale, num_feat=64)
        self.harmonization_net = HAMO_MSRResNet(num_block=4, num_in_ch=num_in_ch + num_in_ch_local, num_out_ch=num_out_ch, upscale=1, num_feat=64)

        self.rroi_align = ROIAlignRotated(self.rroi_scale, spatial_scale=1, sampling_ratio=1)
        self.rroi_align_gt = ROIAlignRotated((self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale), spatial_scale=1, sampling_ratio=1) #(self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale)
        # self.rroi_align_inverse = ROIAlignRotatedInverse((self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale), spatial_scale=1, sampling_ratio=1)

        # self.alphabet = "_" + self.alphabet

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_out_fuse = nn.Conv2d(num_feat + self.num_out_ch, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)

        self.internal = None

        self.sigmoid = torch.nn.Sigmoid()

        self.iter = 0

    def rroi_reverse_per_im(self, cropped_f, rlines):
        # global_size: (H, W)
        # rlines (x, y, w, h, theta)

        device = cropped_f.device

        N, C, cropped_h, cropped_w = cropped_f.shape[:]
        new_ctr_x, new_ctr_y = cropped_w / 2, cropped_w / 2

        pad_size = (cropped_w - cropped_h) // 2

        torch_image_batches = torch.nn.functional.pad(cropped_f, (0, 0, pad_size, pad_size))

        # print("torch_image_batches:", cropped_f.shape, torch_image_batches.shape)

        x_ctr, y_ctr, width, height, angle = rlines[:, 1:2], rlines[:, 2:3], rlines[:, 3:4], rlines[:, 4:5], rlines[:, 5:6]

        arc_batches = angle / 180.0 * 3.141592653589793 ############ clock-wise

        a11, a12, a21, a22 = torch.cos(arc_batches), \
                             -torch.sin(arc_batches), \
                             torch.sin(arc_batches), \
                             torch.cos(arc_batches)

        reshape_w_factor = width / cropped_w
        reshape_h_factor = height / cropped_h

        line = torch.linspace(-1, 1, cropped_w)
        meshx, meshy = torch.meshgrid((line, line))
        meshx = meshx.to(device)
        meshy = meshy.to(device)

        # print("mesh:", meshx.shape, meshy.shape)

        # [1, H, W, 1] / [N, 1, 1, 1] -> [N, H, W, 1]
        # meshx_resize = (meshx + new_ctr_x)[None, ..., None] / reshape_w_factor[:, None, None, ...] - new_ctr_x
        # meshy_resize = (meshy + new_ctr_y)[None, ..., None] / reshape_h_factor[:, None, None, ...] - new_ctr_y
        # print("width:", width, height, reshape_w_factor, reshape_h_factor)
        meshx_resize = (meshx)[None, ..., None] / (reshape_h_factor[:, None, None, ...] + 1e-5)
        meshy_resize = (meshy)[None, ..., None] / (reshape_w_factor[:, None, None, ...] + 1e-5)

        # print("meshy_resize:", meshy_resize.shape, meshx_resize.shape, meshy_resize)
        grid = torch.cat((meshy_resize, meshx_resize), -1)
        # grid = grid.unsqueeze(0)
        resize_warped = F.grid_sample(torch_image_batches, grid, mode='bicubic', align_corners=False)

        x_shift = x_ctr - new_ctr_x
        y_shift = y_ctr - new_ctr_y

        affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1), torch.zeros_like(a11).to(device).unsqueeze(1),
                                   a21.unsqueeze(1), a22.unsqueeze(1), torch.zeros_like(a11).to(device).unsqueeze(1)], dim=1)
        affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

        affine_grid = F.affine_grid(affine_matrix, [N, C, cropped_w, cropped_w])
        rotated_patches = F.grid_sample(resize_warped, affine_grid, align_corners=False)

        # meshx, meshy = torch.meshgrid(torch_image_batches.shape[-2:])
        meshx_shift = meshx[None, ..., None] - y_shift[:, None, None, ...] / new_ctr_y
        meshy_shift = meshy[None, ..., None] - x_shift[:, None, None, ...] / new_ctr_y
        grid = torch.cat((meshy_shift, meshx_shift), -1)
        inversed_patches = F.grid_sample(rotated_patches, grid, mode='bilinear', align_corners=False)

        return inversed_patches


    def find_rlines(self, text_mask):

        device = text_mask.device

        text_mask_np = text_mask.data.cpu().numpy().astype(np.uint8)
        N, C, H, W = text_mask_np.shape
        rlines = []
        for idx in range(N):
            text_mask_i = text_mask_np[idx][0]
            # print("text_mask_i:", text_mask_i.shape, np.unique(text_mask_i))
            contours, _ = cv2.findContours(text_mask_i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                width = rect[1][0]
                height = rect[1][1]
                if abs(rect[2]) > 45:
                    # width = rect[1][0]
                    # height = rect[1][1]
                    rect = (rect[0], (height, width), (90 - abs(rect[2])))

                lines.append((rect[0][0], rect[0][1], rect[1][0], (rect[1][1] * 1.2), -rect[2]))
            rlines.append(np.array(lines).astype(np.float32))

            # print("rlines:", rlines)
        return [torch.tensor(lines).float().to(device) for lines in rlines]

    def forward(self, x, y=None, text_mask=None):
        # rlines [(x, y, w, h, theta)]

        self.iter += 1

        # print("y:", len(y), y)
        # print("y:", y.shape)

        x = x[:, :self.num_in_ch]
        if not y is None:
            y = y[:, :self.num_in_ch]

        device = x.device
        N, C, H, W = x.shape
        Nm, Cm, Hm, Wm = text_mask.shape

        if Hm != H or Wm != W:
            lr_textmask = torch.nn.functional.interpolate(text_mask, (H, W), mode="nearest")
        else:
            lr_textmask = text_mask
        rlines = self.find_rlines(lr_textmask)
        gt_rlines = [] #self.find_rlines(text_mask)
        for lines in rlines:
            # lines
            gt_lines = lines
            if len(lines) > 0:
                # lines[:, :4] = lines[:, :4] * self.upscale
                factor = torch.tensor([self.upscale, self.upscale, self.upscale, self.upscale, 1]).to(lines.device)
                gt_lines = lines * factor.reshape(1, -1)
            gt_rlines.append(gt_lines)
        # tm_np = text_mask.data.cpu().numpy()
        # print("text_mask:", text_mask.shape, np.unique(tm_np))

        cropped_xs = []
        cropped_ys = []
        for idx in range(N):
            lines = rlines[idx]
            b_num = lines.shape[0]
            if b_num > 0:
                order = torch.zeros((b_num, 1)).to(device)
                lines = torch.cat([order, lines], dim=1)
                # print("lines:", lines.shape, lines)
                cropped_xs.append(self.rroi_align(x[idx:idx+1], lines))
            else:
                cropped_xs.append(torch.zeros(1, C, self.rroi_scale[0], self.rroi_scale[1]).to(device) + 2e-10)
            # if self.training:
            if not y is None:
                if b_num > 0:
                    order = torch.zeros((b_num, 1)).to(device)
                    # print("b_num:", b_num, len(gt_rlines[idx]))
                    gt_lines = torch.cat([order, gt_rlines[idx]], dim=1)
                    # print("gt_rlines[idx]:", gt_rlines[idx])
                    # print("scale:", (self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale))
                    cropped_ys.append(self.rroi_align_gt(y[idx:idx+1], gt_lines))
                else:
                    cropped_ys.append(torch.zeros(1, C, self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale).to(device) + 2e-10)

        cropped_srs = []
        cropped_sr_fs = []
        global_canvas = []
        # print("rlines:", len(rlines), len(gt_rlines))
        # print("cropped_xs:", len(cropped_xs), len(cropped_ys))

        x_rec_priors = []
        y_rec_priors = []

        for idx in range(N):

            cropped_x = cropped_xs[idx]
            cropped_y = cropped_ys[idx]
            if self.training:
                cropped_sr, x_rec, y_rec = self.text_recovery_net(cropped_x, cropped_y)
            else:
                cropped_sr = self.text_recovery_net(cropped_x, cropped_y)
                x_rec = None
                y_rec = None
            x_rec_priors.append(x_rec)
            y_rec_priors.append(y_rec)
            cropped_srs.append(cropped_sr)
            # internal_feature = self.text_recovery_net.get_internal_features()
            # in_C = internal_feature.shape[1]
            lines = gt_rlines[idx]
            b_num = len(lines)

            if b_num > 0:
                order = torch.arange(b_num).to(x.device).reshape(-1, 1)
                lines = torch.cat([order, lines], dim=1)
                # print("lines:", lines)
                # print("internal_feature:", internal_feature.sum())
                # print("inloop_box:", lines)
            else:
                lines = torch.tensor([0, -1, -1, 0, 0, 0]).to(x.device).reshape(-1, 6).float()

            global_canvas_per_im = self.rroi_reverse_per_im(cropped_sr, lines)  # self.rroi_align_inverse(cropped_sr, lines, (lines.shape[0], C, H * self.upscale, W * self.upscale)) # self.rroi_reverse_per_im(internal_feature, (H, W), lines).detach()
            weighted_map = (global_canvas_per_im != 0).float()
            weighted_map = weighted_map.sum(0).unsqueeze(0)
            global_canvas_per_im = global_canvas_per_im.sum(0).unsqueeze(0) / (weighted_map + 1e-10)
            global_canvas.append(global_canvas_per_im)

        # for can in global_canvas:
        #     print("can:", can.shape)

        global_canvas = torch.cat(global_canvas, dim=0)
        out = self.global_sr_net.forward_feature(x)

        # out = torch.cat([global_srout, global_canvas], dim=1) # adain(global_srout, global_canvas)
        # out = self.lrelu(self.conv_out_fuse(out))

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale in [2, 3]:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        # global_canvas = global_canvas / (torch.abs(global_canvas).max() + 1e-10)
        # print("final_out:", out.max(), global_canvas.max())
        # out = torch.cat([out, global_canvas], dim=1) # adain(global_srout, global_canvas)
        # out = self.lrelu(self.conv_out_fuse(out))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        # fused_weighted = torch.zeros_like(global_canvas).to(device)
        # fused_weighted = (global_canvas != 0).float()
        # print("shape:", out.shape, global_canvas.shape)
        # out = global_canvas * fused_weighted + out * (1 - fused_weighted)
        # print("base:", out.shape, base.shape)

        out_all = torch.cat([global_canvas, out], dim=1)

        out_harmonized = self.harmonization_net(out_all)
        # print("out_harmonized:", out_harmonized.shape, base.shape, out.shape)
        out_harmonized += base

        cropped_xs = torch.cat(cropped_xs, dim=0)
        if self.training:
            cropped_ys = torch.cat(cropped_ys, dim=0)
            x_rec_priors = torch.cat(x_rec_priors, dim=0)
            y_rec_priors = torch.cat(y_rec_priors, dim=0)
        cropped_srs = torch.cat(cropped_srs, dim=0)

        if self.iter % 100 == 0 and self.training:
            if not os.path.isdir("internal/"):
                os.makedirs("internal/")
            for i in range(cropped_srs.shape[0]):
                cropped_x = cropped_xs[i][:, :3]
                cropped_sr = cropped_srs[i][:, :3]
                cropped_y = cropped_ys[i][:, :3]

                lq_np = cropped_x.permute(1, 2, 0).data.cpu().numpy()
                sr_np = cropped_sr.permute(1, 2, 0).data.cpu().numpy()
                gt_np = cropped_y.permute(1, 2, 0).data.cpu().numpy()
                # print(lq_np.shape, sr_np.shape, gt_np.shape)
                cv2.imwrite("internal/cropped_" + str(i) + "_lq.jpg", (lq_np * 255).astype(np.uint8))
                cv2.imwrite("internal/cropped_" + str(i) + "_sr.jpg", (sr_np * 255).astype(np.uint8))
                cv2.imwrite("internal/cropped_" + str(i) + "_gt.jpg", (gt_np * 255).astype(np.uint8))

        if self.training:
            return out_harmonized, out, global_canvas, cropped_srs, cropped_xs, cropped_ys, x_rec_priors, y_rec_priors
        else:
            return out_harmonized, out, global_canvas, cropped_srs, cropped_xs, None, x_rec_priors, y_rec_priors



@ARCH_REGISTRY.register()
class TE_STISRv2(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_in_ch_local=3, num_out_ch_local=3, num_feat=64, num_block=16, upscale=4, cropped_upscale=2, out_scale=(256, 256)):
        super(TE_STISRv2, self).__init__()
        self.upscale = upscale
        self.rroi_scale = (32, 128) # (32, 128) #
        self.inverse_rroi_scale = (out_scale[0]//upscale, out_scale[1]//upscale)

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.cropped_upscale = cropped_upscale
        self.num_out_ch_local = num_out_ch_local

        self.text_recovery_net = CLIP_RRDBNet(num_in_ch=num_in_ch_local,
                                            num_out_ch=num_out_ch_local,
                                            scale=cropped_upscale,
                                            num_feat=num_feat,
                                            num_block=23,
                                            num_grow_ch=32) # TP_MSRResNet(num_block=8, upscale=cropped_upscale, num_feat=64)
        # self.global_sr_net = MSRResNet(num_in_ch=num_in_ch+num_out_ch_local * self.upscale ** 2, num_out_ch=num_out_ch,
        #                                            num_block=num_block,
        #                                            upscale=upscale, num_feat=num_feat)

        #self.text_recovery_net = Restormer(
        #    inp_channels=num_in_ch_local,
        #    out_channels=num_out_ch_local,
        #    dim=24,
        #    num_blocks=[4, 6, 6, 8],
        #    num_refinement_blocks=4,
        #    heads=[1, 2, 4, 8],
        #    bias=False,
        #    LayerNorm_type='BiasFree',  ## Other option 'BiasFree'
        #    dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
        #)

        # self.text_recovery_net = SCUNet_v2(in_nc=num_in_ch_local, config=[2, 2, 2, 2, 2, 2, 2], dim=64, drop_path_rate=0.0, input_resolution=256)


        self.global_sr_net = RRDBNet(num_in_ch=num_in_ch+num_out_ch_local * self.upscale ** 2,
                                     num_out_ch=num_out_ch,
                                     scale=upscale,
                                     num_feat=64,
                                     num_block=23,
                                     num_grow_ch=32)

        self.rroi_align = ROIAlignRotated(self.rroi_scale, spatial_scale=1, sampling_ratio=1)
        self.rroi_align_gt = ROIAlignRotated((self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale), spatial_scale=1, sampling_ratio=1) #(self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale)

        '''
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)
        
        self.conv_out_fuse = nn.Conv2d(num_feat + self.num_out_ch, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)
        '''
        self.internal = None

        self.sigmoid = torch.nn.Sigmoid()

        self.iter = 0

    def rroi_reverse_per_im(self, cropped_f, rlines):
        # global_size: (H, W)
        # rlines (x, y, w, h, theta)

        device = cropped_f.device

        N, C, cropped_h, cropped_w = cropped_f.shape[:]
        new_ctr_x, new_ctr_y = cropped_w / 2, cropped_w / 2

        pad_size = (cropped_w - cropped_h) // 2

        torch_image_batches = torch.nn.functional.pad(cropped_f, (0, 0, pad_size, pad_size))

        # print("torch_image_batches:", cropped_f.shape, torch_image_batches.shape)

        x_ctr, y_ctr, width, height, angle = rlines[:, 1:2], rlines[:, 2:3], rlines[:, 3:4], rlines[:, 4:5], rlines[:, 5:6]

        arc_batches = angle / 180.0 * 3.141592653589793 ############ clock-wise

        a11, a12, a21, a22 = torch.cos(arc_batches), \
                             -torch.sin(arc_batches), \
                             torch.sin(arc_batches), \
                             torch.cos(arc_batches)

        reshape_w_factor = width / cropped_w
        reshape_h_factor = height / cropped_h

        line = torch.linspace(-1, 1, cropped_w)
        meshx, meshy = torch.meshgrid((line, line))
        meshx = meshx.to(device)
        meshy = meshy.to(device)

        # print("mesh:", meshx.shape, meshy.shape)

        # [1, H, W, 1] / [N, 1, 1, 1] -> [N, H, W, 1]
        # meshx_resize = (meshx + new_ctr_x)[None, ..., None] / reshape_w_factor[:, None, None, ...] - new_ctr_x
        # meshy_resize = (meshy + new_ctr_y)[None, ..., None] / reshape_h_factor[:, None, None, ...] - new_ctr_y
        # print("width:", width, height, reshape_w_factor, reshape_h_factor)
        meshx_resize = (meshx)[None, ..., None] / (reshape_h_factor[:, None, None, ...] + 1e-5)
        meshy_resize = (meshy)[None, ..., None] / (reshape_w_factor[:, None, None, ...] + 1e-5)

        # print("meshy_resize:", meshy_resize.shape, meshx_resize.shape, meshy_resize)
        grid = torch.cat((meshy_resize, meshx_resize), -1)
        # grid = grid.unsqueeze(0)
        resize_warped = F.grid_sample(torch_image_batches, grid, mode='bicubic', align_corners=False)

        x_shift = x_ctr - new_ctr_x
        y_shift = y_ctr - new_ctr_y

        affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1), torch.zeros_like(a11).to(device).unsqueeze(1),
                                   a21.unsqueeze(1), a22.unsqueeze(1), torch.zeros_like(a11).to(device).unsqueeze(1)], dim=1)
        affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

        affine_grid = F.affine_grid(affine_matrix, [N, C, cropped_w, cropped_w])
        rotated_patches = F.grid_sample(resize_warped, affine_grid, align_corners=False)

        # meshx, meshy = torch.meshgrid(torch_image_batches.shape[-2:])
        meshx_shift = meshx[None, ..., None] - y_shift[:, None, None, ...] / new_ctr_y
        meshy_shift = meshy[None, ..., None] - x_shift[:, None, None, ...] / new_ctr_y
        grid = torch.cat((meshy_shift, meshx_shift), -1)
        inversed_patches = F.grid_sample(rotated_patches, grid, mode='bilinear', align_corners=False)

        return inversed_patches


    def find_rlines(self, text_mask):

        device = text_mask.device

        text_mask_np = text_mask.data.cpu().numpy().astype(np.uint8)
        N, C, H, W = text_mask_np.shape
        rlines = []
        for idx in range(N):
            text_mask_i = text_mask_np[idx][0]
            # print("text_mask_i:", text_mask_i.shape, np.unique(text_mask_i))
            contours, _ = cv2.findContours(text_mask_i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                width = rect[1][0]
                height = rect[1][1]
                if abs(rect[2]) < 45:
                    height /= 0.75
                else:
                    width /= 0.75
                if abs(rect[2]) > 45:
                    width = rect[1][0]
                    height = rect[1][1]
                    rect = (rect[0], (height, width), -(90 - abs(rect[2])))
                lines.append((rect[0][0], rect[0][1], rect[1][0], (rect[1][1] * 1.2), -rect[2]))
            if len(lines) == 1:
                lines.append([-1, -1, 0, 0, 0])
            rlines.append(np.array(lines).astype(np.float32))

            # print("rlines:", rlines)
        return [torch.tensor(lines).float().to(device) for lines in rlines]

    def forward(self, x, y=None, text_mask=None):
        # rlines [(x, y, w, h, theta)]

        self.iter += 1

        # print("y:", len(y), y)
        # print("y:", y.shape)

        x = x[:, :self.num_in_ch]
        if not y is None:
            y = y[:, :self.num_in_ch]

        device = x.device
        N, C, H, W = x.shape
        Nm, Cm, Hm, Wm = text_mask.shape

        if Hm != H or Wm != W:
            lr_textmask = torch.nn.functional.interpolate(text_mask, (H, W), mode="nearest")
        else:
            lr_textmask = text_mask
        rlines = self.find_rlines(lr_textmask)
        gt_rlines = [] #self.find_rlines(text_mask)
        for lines in rlines:
            # lines
            gt_lines = lines
            if len(lines) > 0:
                # lines[:, :4] = lines[:, :4] * self.upscale
                factor = torch.tensor([self.upscale, self.upscale, self.upscale, self.upscale, 1]).to(lines.device)
                gt_lines = lines * factor.reshape(1, -1)
            gt_rlines.append(gt_lines)
        # tm_np = text_mask.data.cpu().numpy()
        # print("text_mask:", text_mask.shape, np.unique(tm_np))

        cropped_xs = []
        cropped_ys = []
        for idx in range(N):
            lines = rlines[idx]
            b_num = lines.shape[0]
            if b_num > 0:
                order = torch.zeros((b_num, 1)).to(device)
                lines = torch.cat([order, lines], dim=1)
                # print("lines:", lines.shape, lines)
                cropped_xs.append(self.rroi_align(x[idx:idx+1], lines))
            else:
                cropped_xs.append(torch.zeros(2, C, self.rroi_scale[0], self.rroi_scale[1]).to(device) + 2e-10)
            # if self.training:
            if not y is None:
                if b_num > 0:
                    order = torch.zeros((b_num, 1)).to(device)
                    # print("b_num:", b_num, len(gt_rlines[idx]))
                    gt_lines = torch.cat([order, gt_rlines[idx]], dim=1)
                    cropped_ys.append(self.rroi_align_gt(y[idx:idx+1], gt_lines))
                else:
                    cropped_ys.append(torch.zeros(2, C, self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale).to(device) + 2e-10)

        cropped_srs = []
        cropped_sr_fs = []
        global_canvas = []
        # print("rlines:", len(rlines), len(gt_rlines))
        # print("cropped_xs:", len(cropped_xs), len(cropped_ys))

        x_rec_priors = []
        y_rec_priors = []

        for idx in range(N):

            cropped_x = cropped_xs[idx]
            cropped_y = cropped_ys[idx]
            if self.training:
                cropped_sr, x_rec, y_rec = self.text_recovery_net(cropped_x, cropped_y)
            else:
                cropped_sr = self.text_recovery_net(cropped_x, cropped_y)
                x_rec = None
                y_rec = None
            x_rec_priors.append(x_rec)
            y_rec_priors.append(y_rec)
            cropped_srs.append(cropped_sr)
            # internal_feature = self.text_recovery_net.get_internal_features()
            # in_C = internal_feature.shape[1]
            lines = gt_rlines[idx]
            b_num = len(lines)

            if b_num > 0:
                order = torch.arange(b_num).to(x.device).reshape(-1, 1)
                lines = torch.cat([order, lines], dim=1)
                # print("lines:", lines)
                # print("internal_feature:", internal_feature.sum())
                # print("inloop_box:", lines)
            else:
                lines = torch.tensor([[0, -1, -1, 0, 0, 0], [0, -1, -1, 0, 0, 0]]).to(x.device).reshape(-1, 6).float()

            global_canvas_per_im = self.rroi_reverse_per_im(cropped_sr, lines)  # self.rroi_align_inverse(cropped_sr, lines, (lines.shape[0], C, H * self.upscale, W * self.upscale)) # self.rroi_reverse_per_im(internal_feature, (H, W), lines).detach()
            weighted_map = (global_canvas_per_im != 0).float()
            weighted_map = weighted_map.sum(0).unsqueeze(0)
            global_canvas_per_im = global_canvas_per_im.sum(0).unsqueeze(0) / (weighted_map + 1e-10)
            global_canvas.append(global_canvas_per_im)

        # for can in global_canvas:
        #     print("can:", can.shape)

        global_canvas = torch.cat(global_canvas, dim=0)
        N_g, C_g, H_g, W_g = global_canvas.shape
        global_canvas_stack = global_canvas.reshape(N_g, C_g, H_g//self.upscale, self.upscale, W_g//self.upscale, self.upscale)
        global_canvas_stack = global_canvas_stack.permute(0, 1, 3, 5, 2, 4)
        global_canvas_stack = global_canvas_stack.reshape(N_g, C_g * self.upscale**2, H_g//self.upscale, W_g//self.upscale)

        # global_canvas_base = F.interpolate(global_canvas, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x_all = torch.cat([x, global_canvas_stack.detach()], dim=1)
        out_harmonized = self.global_sr_net(x_all)
        '''
        weighted_map_harm = torch.ones_like(out_harmonized).float()
        weighted_map_canvas = (global_canvas != 0).float()

        if self.num_out_ch_local < self.num_out_ch:
            dim_diff = self.num_out_ch - self.num_out_ch_local
            weight_dim_diff = torch.zeros_like(out_harmonized)[:, :dim_diff]
            weighted_map_canvas = torch.cat([weighted_map_canvas, weight_dim_diff], dim=1)
            global_canvas = torch.cat([global_canvas, weight_dim_diff], dim=1)

        out_harmonized = (out_harmonized + global_canvas) / (weighted_map_harm + weighted_map_canvas)
        '''
        cropped_xs = torch.cat(cropped_xs, dim=0)
        if self.training:
            cropped_ys = torch.cat(cropped_ys, dim=0)
            # x_rec_priors = torch.cat(x_rec_priors, dim=0)
            # y_rec_priors = torch.cat(y_rec_priors, dim=0)
            pass
        cropped_srs = torch.cat(cropped_srs, dim=0)

        if self.iter % 100 == 0 and self.training:
            if not os.path.isdir("internal/"):
                os.makedirs("internal/")
            for i in range(cropped_srs.shape[0]):
                cropped_x = cropped_xs[i][:, :3]
                cropped_sr = cropped_srs[i][:, :3]
                cropped_y = cropped_ys[i][:, :3]

                lq_np = cropped_x.permute(1, 2, 0).data.cpu().numpy()
                sr_np = cropped_sr.permute(1, 2, 0).data.cpu().numpy()
                gt_np = cropped_y.permute(1, 2, 0).data.cpu().numpy()
                # print(lq_np.shape, sr_np.shape, gt_np.shape)
                cv2.imwrite("internal/cropped_" + str(i) + "_lq.jpg", (lq_np * 255).astype(np.uint8))
                cv2.imwrite("internal/cropped_" + str(i) + "_sr.jpg", (sr_np * 255).astype(np.uint8))
                cv2.imwrite("internal/cropped_" + str(i) + "_gt.jpg", (gt_np * 255).astype(np.uint8))

        if self.training:
            return out_harmonized, None, global_canvas, cropped_srs, cropped_xs, cropped_ys, x_rec_priors, y_rec_priors
        else:
            return out_harmonized, None, global_canvas, cropped_srs, cropped_xs, None, x_rec_priors, y_rec_priors


@ARCH_REGISTRY.register()
class TE_STISRv3(nn.Module):
    """Modified SRResNet.

    A compacted version modified from SRResNet in
    "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"
    It uses residual blocks without BN, similar to EDSR.
    Currently, it supports x2, x3 and x4 upsampling scale factor.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        num_block (int): Block number in the body network. Default: 16.
        upscale (int): Upsampling factor. Support x2, x3 and x4. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_in_ch_local=3, num_out_ch_local=3, num_feat=64, num_block=16,
                 upscale=4, cropped_upscale=2, out_scale=(256, 256)):
        super(TE_STISRv3, self).__init__()
        self.upscale = upscale
        self.rroi_scale = (32, 128)  #
        self.inverse_rroi_scale = (out_scale[0] // upscale, out_scale[1] // upscale)

        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.cropped_upscale = cropped_upscale
        self.num_out_ch_local = num_out_ch_local

        self.text_recovery_net = TP_RRDBNetV2(num_in_ch=num_in_ch_local,
                                            num_out_ch=num_out_ch_local,
                                            scale=cropped_upscale,
                                            num_feat=num_feat,
                                            num_block=23,
                                            num_grow_ch=32)  # TP_MSRResNet(num_block=8, upscale=cropped_upscale, num_feat=64)
        # self.global_sr_net = MSRResNet(num_in_ch=num_in_ch+num_out_ch_local * self.upscale ** 2, num_out_ch=num_out_ch,
        #                                            num_block=num_block,
        #                                            upscale=upscale, num_feat=num_feat)

        self.global_sr_net = RRDBNet(num_in_ch=num_in_ch + num_out_ch_local * self.upscale ** 2,
                                     num_out_ch=num_out_ch,
                                     scale=upscale,
                                     num_feat=64,
                                     num_block=23,
                                     num_grow_ch=32)

        self.rroi_align = ROIAlignRotated(self.rroi_scale, spatial_scale=1, sampling_ratio=1)
        self.rroi_align_gt = ROIAlignRotated(
            (self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale), spatial_scale=1,
            sampling_ratio=1)
        '''
        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * self.upscale * self.upscale, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(self.upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_out_fuse = nn.Conv2d(num_feat + self.num_out_ch, num_feat, 3, 1, 1)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.upconv1, self.conv_hr, self.conv_last], 0.1)
        if self.upscale == 4:
            default_init_weights(self.upconv2, 0.1)
        '''
        self.internal = None

        self.sigmoid = torch.nn.Sigmoid()

        self.iter = 0

    def rroi_reverse_per_im(self, cropped_f, rlines):
        # global_size: (H, W)
        # rlines (x, y, w, h, theta)

        device = cropped_f.device

        N, C, cropped_h, cropped_w = cropped_f.shape[:]
        new_ctr_x, new_ctr_y = cropped_w / 2, cropped_w / 2

        pad_size = (cropped_w - cropped_h) // 2

        torch_image_batches = torch.nn.functional.pad(cropped_f, (0, 0, pad_size, pad_size))

        # print("torch_image_batches:", cropped_f.shape, torch_image_batches.shape)

        x_ctr, y_ctr, width, height, angle = rlines[:, 1:2], rlines[:, 2:3], rlines[:, 3:4], rlines[:, 4:5], rlines[:,
                                                                                                             5:6]

        arc_batches = angle / 180.0 * 3.141592653589793  ############ clock-wise

        a11, a12, a21, a22 = torch.cos(arc_batches), \
                             -torch.sin(arc_batches), \
                             torch.sin(arc_batches), \
                             torch.cos(arc_batches)

        reshape_w_factor = width / cropped_w
        reshape_h_factor = height / cropped_h

        line = torch.linspace(-1, 1, cropped_w)
        meshx, meshy = torch.meshgrid((line, line))
        meshx = meshx.to(device)
        meshy = meshy.to(device)

        # print("mesh:", meshx.shape, meshy.shape)

        # [1, H, W, 1] / [N, 1, 1, 1] -> [N, H, W, 1]
        # meshx_resize = (meshx + new_ctr_x)[None, ..., None] / reshape_w_factor[:, None, None, ...] - new_ctr_x
        # meshy_resize = (meshy + new_ctr_y)[None, ..., None] / reshape_h_factor[:, None, None, ...] - new_ctr_y
        # print("width:", width, height, reshape_w_factor, reshape_h_factor)
        meshx_resize = (meshx)[None, ..., None] / (reshape_h_factor[:, None, None, ...] + 1e-5)
        meshy_resize = (meshy)[None, ..., None] / (reshape_w_factor[:, None, None, ...] + 1e-5)

        # print("meshy_resize:", meshy_resize.shape, meshx_resize.shape, meshy_resize)
        grid = torch.cat((meshy_resize, meshx_resize), -1)
        # grid = grid.unsqueeze(0)
        resize_warped = F.grid_sample(torch_image_batches, grid, mode='bicubic', align_corners=False)

        x_shift = x_ctr - new_ctr_x
        y_shift = y_ctr - new_ctr_y

        affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1), torch.zeros_like(a11).to(device).unsqueeze(1),
                                   a21.unsqueeze(1), a22.unsqueeze(1), torch.zeros_like(a11).to(device).unsqueeze(1)],
                                  dim=1)
        affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

        affine_grid = F.affine_grid(affine_matrix, [N, C, cropped_w, cropped_w])
        rotated_patches = F.grid_sample(resize_warped, affine_grid, align_corners=False)

        # meshx, meshy = torch.meshgrid(torch_image_batches.shape[-2:])
        meshx_shift = meshx[None, ..., None] - y_shift[:, None, None, ...] / new_ctr_y
        meshy_shift = meshy[None, ..., None] - x_shift[:, None, None, ...] / new_ctr_y
        grid = torch.cat((meshy_shift, meshx_shift), -1)
        inversed_patches = F.grid_sample(rotated_patches, grid, mode='bilinear', align_corners=False)

        return inversed_patches

    def find_rlines(self, text_mask):

        device = text_mask.device

        text_mask_np = text_mask.data.cpu().numpy().astype(np.uint8)
        N, C, H, W = text_mask_np.shape
        rlines = []
        for idx in range(N):
            text_mask_i = text_mask_np[idx][0]
            # print("text_mask_i:", text_mask_i.shape, np.unique(text_mask_i))
            contours, _ = cv2.findContours(text_mask_i, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            lines = []
            for cnt in contours:
                rect = cv2.minAreaRect(cnt)
                width = rect[1][0]
                height = rect[1][1]
                if abs(rect[2]) < 45:
                    height /= 0.75
                else:
                    width /= 0.75
                if abs(rect[2]) > 45:
                    width = rect[1][0]
                    height = rect[1][1]
                    rect = (rect[0], (height, width), -(90 - abs(rect[2])))
                lines.append((rect[0][0], rect[0][1], rect[1][0], (rect[1][1] * 1.2), -rect[2]))
            rlines.append(np.array(lines).astype(np.float32))

            # print("rlines:", rlines)
        return [torch.tensor(lines).float().to(device) for lines in rlines]

    def forward(self, x, y=None, text_mask=None):
        # rlines [(x, y, w, h, theta)]

        self.iter += 1

        # print("y:", len(y), y)
        # print("y:", y.shape)

        x = x[:, :self.num_in_ch]
        if not y is None:
            y = y[:, :self.num_in_ch]

        device = x.device
        N, C, H, W = x.shape
        Nm, Cm, Hm, Wm = text_mask.shape

        if Hm != H or Wm != W:
            lr_textmask = torch.nn.functional.interpolate(text_mask, (H, W), mode="nearest")
        else:
            lr_textmask = text_mask
        rlines = self.find_rlines(lr_textmask)
        gt_rlines = []  # self.find_rlines(text_mask)
        for lines in rlines:
            # lines
            gt_lines = lines
            if len(lines) > 0:
                # lines[:, :4] = lines[:, :4] * self.upscale
                factor = torch.tensor([self.upscale, self.upscale, self.upscale, self.upscale, 1]).to(lines.device)
                gt_lines = lines * factor.reshape(1, -1)
            gt_rlines.append(gt_lines)
        # tm_np = text_mask.data.cpu().numpy()
        # print("text_mask:", text_mask.shape, np.unique(tm_np))

        cropped_xs = []
        cropped_ys = []
        for idx in range(N):
            lines = rlines[idx]
            b_num = lines.shape[0]
            if b_num > 0:
                order = torch.zeros((b_num, 1)).to(device)
                lines = torch.cat([order, lines], dim=1)
                # print("lines:", lines.shape, lines)
                cropped_xs.append(self.rroi_align(x[idx:idx + 1], lines))
            else:
                cropped_xs.append(torch.zeros(1, C, self.rroi_scale[0], self.rroi_scale[1]).to(device) + 2e-10)
            # if self.training:
            if not y is None:
                if b_num > 0:
                    order = torch.zeros((b_num, 1)).to(device)
                    # print("b_num:", b_num, len(gt_rlines[idx]))
                    gt_lines = torch.cat([order, gt_rlines[idx]], dim=1)
                    # print("gt_rlines[idx]:", gt_rlines[idx])
                    # print("scale:", (self.rroi_scale[0] * self.cropped_upscale, self.rroi_scale[1] * self.cropped_upscale))
                    cropped_ys.append(self.rroi_align_gt(y[idx:idx + 1], gt_lines))
                else:
                    cropped_ys.append(torch.zeros(1, C, self.rroi_scale[0] * self.cropped_upscale,
                                                  self.rroi_scale[1] * self.cropped_upscale).to(device) + 2e-10)

        cropped_srs = []
        cropped_sr_fs = []
        global_canvas = []

        x_rec_priors = []
        y_rec_priors = []

        for idx in range(N):

            cropped_x = cropped_xs[idx]
            cropped_y = cropped_ys[idx]
            if self.training:
                cropped_sr, x_rec, y_rec = self.text_recovery_net(cropped_x, cropped_y)
            else:
                cropped_sr = self.text_recovery_net(cropped_x, cropped_y)
                x_rec = None
                y_rec = None
            x_rec_priors.append(x_rec)
            y_rec_priors.append(y_rec)
            cropped_srs.append(cropped_sr)
            # internal_feature = self.text_recovery_net.get_internal_features()
            # in_C = internal_feature.shape[1]
            lines = gt_rlines[idx]
            b_num = len(lines)

            if b_num > 0:
                order = torch.arange(b_num).to(x.device).reshape(-1, 1)
                lines = torch.cat([order, lines], dim=1)
                # print("lines:", lines)
                # print("internal_feature:", internal_feature.sum())
                # print("inloop_box:", lines)
            else:
                lines = torch.tensor([0, -1, -1, 0, 0, 0]).to(x.device).reshape(-1, 6).float()

            global_canvas_per_im = self.rroi_reverse_per_im(cropped_sr,
                                                            lines)  # self.rroi_align_inverse(cropped_sr, lines, (lines.shape[0], C, H * self.upscale, W * self.upscale)) # self.rroi_reverse_per_im(internal_feature, (H, W), lines).detach()
            weighted_map = (global_canvas_per_im != 0).float()
            weighted_map = weighted_map.sum(0).unsqueeze(0)
            global_canvas_per_im = global_canvas_per_im.sum(0).unsqueeze(0) / (weighted_map + 1e-10)
            global_canvas.append(global_canvas_per_im)

        # for can in global_canvas:
        #     print("can:", can.shape)

        global_canvas = torch.cat(global_canvas, dim=0)
        N_g, C_g, H_g, W_g = global_canvas.shape
        global_canvas_stack = global_canvas.reshape(N_g, C_g, H_g // self.upscale, self.upscale, W_g // self.upscale,
                                                    self.upscale)
        global_canvas_stack = global_canvas_stack.permute(0, 1, 3, 5, 2, 4)
        global_canvas_stack = global_canvas_stack.reshape(N_g, C_g * self.upscale ** 2, H_g // self.upscale,
                                                          W_g // self.upscale)

        # global_canvas_base = F.interpolate(global_canvas, (x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        x_all = torch.cat([x, global_canvas_stack.detach()], dim=1)
        out_harmonized = self.global_sr_net(x_all)
        '''
        weighted_map_harm = torch.ones_like(out_harmonized).float()
        weighted_map_canvas = (global_canvas != 0).float()

        if self.num_out_ch_local < self.num_out_ch:
            dim_diff = self.num_out_ch - self.num_out_ch_local
            weight_dim_diff = torch.zeros_like(out_harmonized)[:, :dim_diff]
            weighted_map_canvas = torch.cat([weighted_map_canvas, weight_dim_diff], dim=1)
            global_canvas = torch.cat([global_canvas, weight_dim_diff], dim=1)

        out_harmonized = (out_harmonized + global_canvas) / (weighted_map_harm + weighted_map_canvas)
        '''
        cropped_xs = torch.cat(cropped_xs, dim=0)
        if self.training:
            cropped_ys = torch.cat(cropped_ys, dim=0)
            x_rec_priors = torch.cat(x_rec_priors, dim=0)
            y_rec_priors = torch.cat(y_rec_priors, dim=0)
        cropped_srs = torch.cat(cropped_srs, dim=0)

        if self.iter % 100 == 0 and self.training:
            if not os.path.isdir("internal/"):
                os.makedirs("internal/")
            for i in range(cropped_srs.shape[0]):
                cropped_x = cropped_xs[i][:, :3]
                cropped_sr = cropped_srs[i][:, :3]
                cropped_y = cropped_ys[i][:, :3]

                lq_np = cropped_x.permute(1, 2, 0).data.cpu().numpy()
                sr_np = cropped_sr.permute(1, 2, 0).data.cpu().numpy()
                gt_np = cropped_y.permute(1, 2, 0).data.cpu().numpy()
                # print(lq_np.shape, sr_np.shape, gt_np.shape)
                cv2.imwrite("internal/cropped_" + str(i) + "_lq.jpg", (lq_np * 255).astype(np.uint8))
                cv2.imwrite("internal/cropped_" + str(i) + "_sr.jpg", (sr_np * 255).astype(np.uint8))
                cv2.imwrite("internal/cropped_" + str(i) + "_gt.jpg", (gt_np * 255).astype(np.uint8))

        if self.training:
            return out_harmonized, None, global_canvas, cropped_srs, cropped_xs, cropped_ys, x_rec_priors, y_rec_priors
        else:
            return out_harmonized, None, global_canvas, cropped_srs, cropped_xs, None, x_rec_priors, y_rec_priors

