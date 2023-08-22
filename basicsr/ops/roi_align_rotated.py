# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from basicsr.ops import _C


class _ROIAlignRotated(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        output = _C.roi_align_rotated_forward(
            input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_rotated_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
        )
        return grad_input, None, None, None, None, None


roi_align_rotated = _ROIAlignRotated.apply


class ROIAlignRotated(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.

        Note:
            ROIAlignRotated supports continuous coordinate by default:
            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5).
        """
        super(ROIAlignRotated, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx6 boxes. First column is the index into N.
                The other 5 columns are (x_ctr, y_ctr, width, height, angle_degrees).
        """
        assert rois.dim() == 2 and rois.size(1) == 6
        return roi_align_rotated(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ")"
        return tmpstr

if __name__ == "__main__":
    import os

    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import cv2
    import numpy as np
    from PIL import Image

    # from torch.autograd import Variable
    # from rroi_pooling.modules.rroi_pool import RRoIPool

    # import network
    imname = 'gt_show.jpg'

    im = cv2.imread(imname)

    cv2.line(im, (1353, 500), (1000, 853), (255, 0, 255), 3)
    cv2.line(im, (1000, 853), (1000 - 353, 500), (255, 0, 255), 3)
    cv2.line(im, (1000 - 353, 500), (1000, 500 - 353), (255, 0, 255), 3)
    cv2.line(im, (1000, 500 - 353), (1353, 500), (255, 0, 255), 3)
    cv2.line(im, (1250, 500), (1000, 750), (255, 0, 255), 3)
    cv2.line(im, (1000, 750), (750, 500), (255, 0, 255), 3)
    cv2.line(im, (750, 500), (1000, 250), (255, 0, 255), 3)
    cv2.line(im, (1000, 250), (1250, 500), (255, 0, 255), 3)

    # cv2.imshow('win1', im.copy())

    ma = np.expand_dims(im, 0).transpose(0, 3, 1, 2)
    iminfo = np.array([im.shape[0], im.shape[1], 1])
    rois = np.array([[0, 1000, 500, 500, 200, 45], [0, 1000, 500, 500, 200, 0], [0, 0, 0, 500, 500, 45]],
                    dtype=np.float32)
    print('ma:', ma.shape, iminfo)

    ma = torch.tensor(ma).float().cuda()
    iminfo = torch.tensor(iminfo).cuda()  # network.np_to_variable(iminfo, is_cuda=True)
    rois = torch.tensor(rois).cuda()  # network.np_to_variable(rois, is_cuda=True)
    print('ma.requires_grad:', ma.requires_grad)
    ma.requires_grad = True
    rroi_pool = ROIAlignRotated((100, 200), 1.0 / 1, 2)

    pooled = rroi_pool(ma, rois)

    print('pooled:', pooled.size())

    crop = pooled.data.cpu().numpy()

    print(crop.shape, np.unique(crop))

    crop_trans = crop.transpose(0, 2, 3, 1)
    crop1 = Image.fromarray(crop_trans[0].astype(np.uint8))
    crop2 = Image.fromarray(crop_trans[1].astype(np.uint8))
    crop3 = Image.fromarray(cv2.cvtColor(crop_trans[2].astype(np.uint8), cv2.COLOR_BGR2RGB))
    crop1.save('crop1.jpg', 'jpeg')
    crop2.save('crop2.jpg', 'jpeg')
    crop3.save('crop3.jpg', 'jpeg')

    print(crop.shape)

    mean = torch.mean(pooled)

    # mean.backward()

    grad = torch.autograd.grad(mean, ma)

    print(
        'grad:',
        type(grad),
        len(grad),
        np.unique(grad[0].data.cpu().numpy()),
        np.where(grad[0].data.cpu().numpy() > 0)
    )
