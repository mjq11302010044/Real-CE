import torch
from torch import nn as nn

from basicsr.archs.arch_util import ResidualBlockNoBN, Upsample, make_layer
from basicsr.utils.registry import ARCH_REGISTRY

class RSB(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 scale
                 ):
        super(RSB, self).__init__()

        self.conv_first = nn.Conv2d(channels, channels, kernel_size, 1, 1)
        self.conv_last = nn.Conv2d(channels, channels, kernel_size, 1, 1)
        self.scale = scale

    def forward(self, x):

        x = self.conv_first(x)
        x = torch.nn.functional.relu(x)
        x = self.conv_last(x)

        x *= self.scale
        return x

'''
# upsampling module
x = tf.layers.conv2d_transpose(input, self.channels, self.up_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)
x = tf.layers.conv2d(x, self.channels, self.up_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)
x = tf.layers.conv2d_transpose(x, self.channels, self.up_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)
# prediction module
x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)
x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)
x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)
x_before = 1 * x
for i in range(self.RSB_num):
    x = RSB(x, self.channels, self.pre_kernel_size, self.RSB_scale)
x_after = 1 * x
x = tf.add(x_before, x_after)
x = tf.layers.conv2d_transpose(x, self.channels, self.pre_kernel_size, (2, 2), padding='same', activation=tf.nn.relu)
x = tf.layers.conv2d(x, self.channels, self.pre_kernel_size, (1, 1), padding='same', activation=tf.nn.relu)
x = tf.layers.conv2d(x, self.EPCNN_out_channels, self.pre_kernel_size, (1, 1), padding='same')
return x
'''

@ARCH_REGISTRY.register()
class EPCNN(nn.Module):
    """EPCNN network structure.

    Paper: Enhanced Deep Residual Networks for Single Image Super-Resolution.
    Ref git repo: https://github.com/thstkdgus35/EDSR-PyTorch

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_block (int): Block number in the trunk network. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 num_feat=64,
                 num_block=16,
                 up_kernel_size=5,
                 pre_kernel_size=3,
                 upscale=4,
                 res_scale=1,
                 img_range=255.,
                 RSB_scale=0.1,
                 rgb_mean=(0.4488, 0.4371, 0.4040)):
        super(EPCNN, self).__init__()

        self.img_range = img_range

        self.mean = torch.Tensor(rgb_mean).view(1, num_in_ch, 1, 1)

        self.conv_up1 = nn.Conv2dTranspose(num_in_ch, num_feat, up_kernel_size, (2, 2), 2)
        self.conv_norm = nn.Conv2d(num_feat, num_feat, up_kernel_size, 1, 2)
        self.conv_up2 = nn.Conv2dTranspose(num_feat, num_feat, up_kernel_size, (2, 2), 2)

        self.conv_pred1 = nn.Conv2d(num_feat, num_feat, pre_kernel_size, 1, 1)
        self.conv_pred2 = nn.Conv2d(num_feat, num_feat, pre_kernel_size, 1, 1)
        self.conv_pred3 = nn.Conv2d(num_feat, num_feat, pre_kernel_size, (2, 2), 1)

        self.RSB_1 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_2 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_3 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_4 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_5 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_6 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_7 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_8 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_9 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_10 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_11 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_12 = RSB(num_feat, pre_kernel_size, RSB_scale)
        self.RSB_13 = RSB(num_feat, pre_kernel_size, RSB_scale)

        self.conv_after1 = nn.Conv2dTranspose(num_feat, num_feat, pre_kernel_size, (2, 2), 1)
        self.conv_after2 = nn.Conv2d(num_feat, num_feat, pre_kernel_size, 1, 1)
        self.conv_after3 = nn.Conv2d(num_feat, num_out_ch, pre_kernel_size, 1, 1)

    def forward(self, x, label_mask=None):
        self.mean = self.mean.type_as(x)

        x = (x - self.mean) * self.img_range

        x = self.conv_up1(x)
        x = self.conv_norm(x)
        x = self.conv_up2(x)

        x = self.conv_pred1(x)
        x = self.conv_pred2(x)
        x = self.conv_pred3(x)

        x_before = x

        x = self.RSB_1(x)
        x = self.RSB_2(x)
        x = self.RSB_3(x)
        x = self.RSB_4(x)
        x = self.RSB_5(x)
        x = self.RSB_6(x)
        x = self.RSB_7(x)
        x = self.RSB_8(x)
        x = self.RSB_9(x)
        x = self.RSB_10(x)
        x = self.RSB_11(x)
        x = self.RSB_12(x)
        x = self.RSB_13(x)

        x = x + x_before

        x = self.conv_after1(x)
        x = self.conv_after2(x)
        x = self.conv_after3(x)

        x = x / self.img_range + self.mean

        return x
