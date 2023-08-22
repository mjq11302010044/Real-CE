from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_region_psnr, calculate_region_ssim, calculate_psnr_torch, calculate_ssim_torch, calculate_lpips
from .recognition import calculate_recognition

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
