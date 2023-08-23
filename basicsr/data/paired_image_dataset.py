import os
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F

import random
import json
import numpy as np
import math
import torch
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop, paired_random_cropwmask, paired_random_cropwmasks
from basicsr.data.real_esrgan_bsrgan_degradation import real_esrgan_degradation, bsrgan_degradation, bsrgan_degradation_plus
from basicsr.data.corpus import alphabet
from basicsr.data import degradations as degradations
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.degradations import blur, noisy, JPEG_compress
from basicsr.data.degradation_v2 import degradation_pipeline

from PIL import Image, ImageFilter, ImageDraw, ImageFont
import imgaug.augmenters as ia
import time, re, sys
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
from basicsr.data.data_util import paths_from_folder
import cv2

LABEL_LIST = open("./basicsr/metrics/benchmark.txt", "r").readlines()[0]
LABEL_LIST = " " + LABEL_LIST.replace("\n", "")
LABEL_DICT = {}
for cnt in range(len(LABEL_LIST)):
    ch = LABEL_LIST[cnt]
    LABEL_DICT[ch] = cnt


def apply_translation(img_lr, img_hr):
    H_l, W_l = img_lr.shape[:2]
    H_h, W_h = img_hr.shape[:2]

    assert H_l == H_h, "Keep input the same..."

    shift_pixel = 4
    shift_x = np.random.randint(0, shift_pixel + 1)
    shift_y = np.random.randint(0, shift_pixel + 1)

    shift_x_l = np.random.randint(0, shift_pixel + 1)
    shift_y_l = np.random.randint(0, shift_pixel + 1)

    # print("img shape:", img_lr.shape[:2], img_hr.shape[:2])

    # translate and re-adjust
    img_hr = img_hr[shift_y:, shift_x:]
    h, w = img_hr.shape[:2]

    if np.random.randint(1, 5) == 1:
        img_lr = img_lr[shift_y_l:shift_y_l+h, shift_x_l:shift_x_l+w]
    elif np.random.randint(1, 5) == 2:
        h_l, w_l = img_lr.shape[:2]
        t = max(h_l - shift_y_l - h, 0)
        img_lr = img_lr[t:h_l - shift_y_l, shift_x_l:shift_x_l+w]
    elif np.random.randint(1, 5) == 3:
        h_l, w_l = img_lr.shape[:2]
        t = max(h_l - shift_y_l - h, 0)
        l = max(w_l - shift_x_l - w, 0)
        img_lr = img_lr[t:h_l - shift_y_l, l:w_l - shift_x_l]
    elif np.random.randint(1, 5) == 4:
        h_l, w_l = img_lr.shape[:2]
        l = max(w_l - shift_x_l - w, 0)
        img_lr = img_lr[shift_y_l:shift_y_l+h, l:w_l - shift_x_l]

    # print("shift:", img_hr.shape[:2], img_lr.shape[:2])

    return img_lr, img_hr

@DATASET_REGISTRY.register()
class PairedImageDatasetRealCE(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCE, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #                                                   self.opt['meta_info_file'], self.filename_tmpl)
        else:
            # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            self.gt_folder = os.path.join(self.gt_folder, self.opt['phase'])

            fl_13mmdir = os.path.join(self.gt_folder, "13mm")
            fl_26mmdir = os.path.join(self.gt_folder, "26mm")
            fl_52mmdir = os.path.join(self.gt_folder, "52mm")

            self.paths = []

            imlist = os.listdir(fl_13mmdir)

            for imname in imlist:
                impath_13mm = os.path.join(fl_13mmdir, imname)
                impath_26mm = os.path.join(fl_26mmdir, imname)
                impath_52mm = os.path.join(fl_52mmdir, imname)

                # self.paths.append({"lq_path": impath_13mm, "gt_path": impath_26mm})
                self.paths.append({"lq_path": impath_13mm, "gt_path": impath_52mm})
                self.paths.append({"lq_path": impath_26mm, "gt_path": impath_52mm})
                self.paths.append({"lq_path": impath_52mm, "gt_path": impath_52mm})

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        # print("img_hr:", H, W)
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)

        # print("img_lq:", img_lq.shape)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



@DATASET_REGISTRY.register()
class TextDegradationDataset(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(TextDegradationDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.dataroot = opt['path_bg']
        self.gt_folder = opt['path_bg']
        # self.mask_folder = opt['maskroot']
        self.mean = opt.get("mean", None)#['mean']
        self.std = opt.get("std", None)
        self.out_height = opt['gt_size'][0]
        self.out_width = opt['gt_size'][1]
        self.scale = opt['scale']
        self.CheckNum = int(opt['check_num'])

        # self.CommonWords = opt['CommonWords']
        self.CommonWords = alphabet

        index_not_chinese = [35, 107, 114, 165, 194, 206, 226, 227, 229, 248, 273, 302, 317, 333, 355, 371, 377, 398,
                             417, 434, 437, 456, 466, 468, 506, 536, 565, 575, 585, 595, 625, 634, 653, 661, 665, 667,
                             732, 756, 815, 841, 920, 935, 949, 957, 1020, 1024, 1028, 1064, 1067, 1133, 1158, 1221,
                             1230, 1258, 1341, 1343, 1369, 1417, 1457, 1496, 1497, 1499, 1517, 1547, 1564, 1600, 1624,
                             1637, 1662, 1693, 1770, 1780, 1802, 1835, 1855, 1857, 1925, 1947, 1958, 1959, 2016, 2031,
                             2034, 2035, 2037, 2059, 2072, 2083, 2089, 2116, 2173, 2217, 2237, 2315, 2330, 2346, 2368,
                             2404, 2435, 2436, 2446, 2451, 2456, 2485, 2488, 2499, 2556, 2557, 2559, 2582, 2583, 2607,
                             2616, 2679, 2680, 2707, 2735, 2748, 2771, 2799, 2865, 2883, 2895, 2913, 2933, 2963, 2991,
                             3063, 3074, 3087, 3110, 3126, 3128, 3129, 3157, 3177, 3178, 3185, 3212, 3213, 3230, 3242,
                             3253, 3258, 3278, 3292, 3303, 3305, 3319, 3367, 3370, 3411, 3437, 3478, 3484, 3486, 3570,
                             3583, 3591, 3612, 3671, 3675, 3708, 3762, 3782, 3790, 3815, 3817, 3836, 3867, 3872, 3906,
                             3913, 3914, 3924, 3955, 3960, 3975, 3993, 4013, 4035, 4041, 4053, 4063, 4068, 4114, 4123,
                             4124, 4166, 4179, 4190, 4216, 4312, 4324, 4333, 4368, 4378, 4431, 4443, 4449, 4458, 4467,
                             4498, 4537, 4552, 4566, 4623, 4637, 4654, 4658, 4668, 4738, 4753, 4784, 4932, 4975, 4979,
                             4983, 5005, 5016, 5026, 5033, 5036, 5051, 5052, 5076, 5082, 5090, 5129, 5135, 5166, 5175,
                             5206, 5243, 5252, 5274, 5288, 5319, 5355, 5370, 5411, 5455, 5468, 5497, 5504, 5654, 5670,
                             5688, 5702, 5734, 5741, 5749, 5812, 5815, 5829, 5842, 5850, 5870, 5880, 5888, 5921, 5955,
                             5957, 5965, 5967, 5994, 5995, 5996, 6021, 6036, 6064, 6071, 6104, 6119, 6166, 6190, 6218,
                             6237, 6251, 6254, 6280, 6286, 6311, 6329, 6344, 6408, 6416, 6445, 6449, 6462, 6472, 6475,
                             6506, 6546, 6567, 6651, 6678, 6723, 6731]
        index_not_chinese = [3074, 3914, 1959, 165, 6064, 3762, 5455, 5370, 4753, 6449, 2607, 4368, 6344, 1064, 2616,
                             1024, 1958, 841, 3278, 5870, 3177, 4449, 5888, 1637, 333, 2059, 5129, 2559, 302, 5076, 434,
                             5670, 2217, 6021, 6445, 2913, 5243, 3790, 2037, 665, 4333, 2034, 2404, 3906, 3671, 5036,
                             4053, 2679, 3486, 6071, 114, 3230]
        english_letter = [3074, 3914, 1959, 165, 6064, 3762, 5455, 5370, 4753, 6449, 2607, 4368, 6344, 1064, 2616, 1024,
                          1958, 841, 3278, 5870, 3177, 4449, 5888, 1637, 333, 2059]
        number_letter = [575, 2116, 1230, 1857, 3157, 1564, 4124, 3708, 2072, 355]
        self.HalfWords = []
        for ind in index_not_chinese:
            self.HalfWords.append(alphabet[ind])
        self.EnglishOnly = []
        self.NumberOnly = []
        for ind in english_letter:
            self.EnglishOnly.append(alphabet[ind])
        for ind in number_letter:
            self.NumberOnly.append(alphabet[ind])
        # self.NumWords = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#？'
        # self.SymbolWords = string.ascii_letters + string.digits + string.punctuation

        with open(opt['corpus_path1'], 'r') as f:
            lines = f.read().split('\n')
        self.corpus1 = [line.replace("\n", "").split("\t")[3].strip() for line in lines if len(line) > opt['min_text_length']]
        self.max_corpus_length = opt['max_text_length']

        with open(opt['corpus_path2'], 'r') as f:
            lines = f.read().split('\n')
        self.corpus1.extend([line.replace("\n", "").split("\t")[0].strip() for line in lines if len(line) > opt['min_text_length']])
        self.max_corpus_length = opt['max_text_length']

        # print("corpus1:", self.corpus1)

        print("[Corpus1] Number of Corpus1 pairs: {}, classes: {}".format(len(self.corpus1), len(self.CommonWords)))

        # self.gray_aug = ia.Grayscale(alpha=random.random())
        self.gray_aug = ia.Grayscale(alpha=random.randint(5, 10) / 10.0)
        error_flg = False
        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(os.path.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        elif self.io_backend_opt['type'] == 'disk':
            self.paths = []
            if os.path.isfile(self.opt['path_bg']):
                fp = open(self.opt['path_bg'], "r")
                lines = fp.read().split("\n")
                lines = [line.strip() for line in lines if len(line)]

                for line in lines:
                    image_file = os.path.join(self.dataroot, line)
                    if not os.path.exists(image_file):
                        print("%s does not exist!" % (image_file))
                        error_flg = True
                    self.paths.append(image_file)
            else:
                self.paths = paths_from_folder(self.gt_folder)

            if error_flg:
                raise ValueError("Some file paths are corrupted! Please re-check your file paths!")
            index = np.arange(len(self.paths))
            np.random.shuffle(index)
            self.paths = np.array(self.paths)
            self.paths = self.paths[index]
            print("[Dataset] Number of Bcakground pairs:", len(self.paths))

            self.font_paths = []
            # Fonts = ['Deng.ttf','Dengb.ttf','Dengl.ttf','simfang.ttf','simhei.ttf','simkai.ttf','simsun.ttc','STSONG.TTF']
            Fonts = os.listdir(opt['path_font'])
            for f in Fonts:
                self.font_paths.append(os.path.join(opt['path_font'], f))

        else:
            # disk backend: scan file list from a folder
            self.paths = paths_from_folder(self.gt_folder)
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def make_dataset(self, dirs):
        images = []
        assert os.path.isdir(dirs), '%s is not a valid directory' % dirs
        for root, _, fnames in sorted(os.walk(dirs)):
            fnames.sort()
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def AddDownSample(self, img):  # downsampling
        sampler = random.randint(15, 35) * 1.0
        h0, w0 = img.shape[:2]
        if random.random() > 0.5:
            img = cv2.resize(img, (int(w0 / sampler * 10.0), int(h0 / sampler * 10.0)), cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (int(w0 / sampler * 10.0), int(h0 / sampler * 10.0)), cv2.INTER_LINEAR)
        out = cv2.resize(img.copy(), (self.out_size // 2, self.out_size // 2), cv2.INTER_LINEAR)
        return out

    def image_add_text(self, img):
        '''
        Get the text
        '''
        text, label = self.get_text()
        while self.check_corpus(text):
            text, label = self.get_text()

        w, h = img.size
        bg_is_white = 0
        if random.random() > 0.5:  # 0.96: #white bg
            img_tmp = Image.new('RGB', (w, h), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            # img_tmp = Image.new('RGB', (w, h), (255,255,255))
            im_numpy = np.array(img_tmp)[:, :, ::-1]  # BGR 0~255
            img_aug = self.gray_aug(image=im_numpy.astype(np.uint8))
            img = Image.fromarray(img_aug[:, :, ::-1])
            img = img_tmp
            bg_is_white = 1

        # 字体的格式 这里的SimHei.ttf需要有这个字体
        fontpath = self.font_paths[random.randint(0, len(self.font_paths) - 1)]
        text_size = random.randint(20, 48)  # if Small > 0.3 else random.randint(128, 235)#235 267 (80, 140)
        x = random.randint(-10, 30)  #
        y = random.randint(-10, 30)  #
        pos = (x, y)  # width, height from top left to bottom right
        fontStyle = ImageFont.truetype(fontpath, text_size, encoding="utf-8")
        mask_color = (255, 255, 255)

        pos_mask = Image.new('L', (w, h), 0)
        drawer_tmp = ImageDraw.Draw(pos_mask)
        char_locs = []
        img_max_width = 0

        for i in range(1, len(text) + 1):
            p = text[:i]
            drawer_tmp.text(pos, p, font=fontStyle, fill=255)
            char_mask = np.array(pos_mask).copy()
            vertical_projection = np.sum(char_mask, axis=0)
            ws = np.where(vertical_projection > 1)  # height, width
            locs = list(ws[0])
            if len(locs) == 0:  # space
                continue
            # if i == 1:
            if len(char_locs) == 0:
                char_locs.append(max(min(locs), 0))
                char_locs.append(min(max(locs), w - 1))
            else:
                new_locs = []
                for l in locs:
                    if l > char_locs[-1]:
                        new_locs.append(l)
                if len(new_locs):
                    char_locs.append(max(min(new_locs), 0))
                    char_locs.append(min(max(new_locs), w - 1))

        if len(char_locs) == 0:
            # print(['error max char_locs', text])
            return None, np.zeros((1, 1)), None, None, None
        img_max_width = max(char_locs)
        if len(text) != len(char_locs) // 2 or len(label) != len(char_locs) // 2 or img_max_width > self.out_height * self.CheckNum:
            # print(['return error inconsistent numbers', text, len(text), len(label), len(char_locs)])
            return None, np.zeros((1, 1)), None, None, None

        if self.CheckNum == 1:
            char_locs = [0, self.out_height]

        for i in range(len(text), self.CheckNum):
            char_locs.append(self.CheckNum * self.out_height)
            char_locs.append(self.CheckNum * self.out_height)

        which_color = random.random()  # common
        if bg_is_white:
            if random.random() > 0.9:
                text_color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))
            else:
                text_color = (0, 0, 0)
        else:
            text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        text_color = (0, 0, 0)
        # if 1: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     pos = (0,0)
        #     text_size = 128
        #     fontpath = self.font_paths[0]
        #     img = Image.new('RGB', (w, h), (255,255,255))
        #     text_color = (0,0,0)

        drawer = ImageDraw.Draw(img)
        drawer.text(pos, text, font=fontStyle, fill=text_color)
        # drawer_mask.text(pos, text, font=fontStyle, fill=mask_color)
        pos_mask = np.array(pos_mask)
        mask = np.repeat(pos_mask[:, :, np.newaxis], 3, axis=2)
        mask[mask > 128] = 255
        mask[mask <= 128] = 0

        img = np.array(img)[:, :, ::-1].astype(np.float32)
        offset_w = min(img_max_width + random.randint(0, 16), 128 * self.CheckNum)
        offset_w = offset_w // 4 * 4

        img = img[:, :offset_w, :]
        mask = mask[:, :offset_w, :]

        if img.shape[-2] < 10 or img.shape[-3] < 10 or img.shape[-2] > 128 * self.CheckNum:
            print('return error 1')
            return None, np.zeros((1, 1)), None, None, None
        return img / 255.0, mask / 255.0, text, label, char_locs

    def read_valid_image(self, gt_path):
        retry = 3
        while retry > 0:
            try:
                # img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)/255.0
                img_bytes = self.file_client.get(gt_path)
                img_gt = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                gt_path = self.paths[random.randint(0, len(self.paths) - 1)]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        return img_gt

    def get_valid_patch(self, gt_path):
        img_gt = self.read_valid_image(gt_path)
        ##flip
        flip = 1 if random.random() > 0.5 else 0
        if flip:
            cv2.flip(img_gt, 1, img_gt)
        random_size = random.randint(320, 400)
        img_gt = cv2.resize(img_gt, (random_size, random_size), cv2.INTER_LINEAR)
        h0, w0 = img_gt.shape[:2]

        h1 = np.random.choice(np.arange(0, h0 // 2))
        w1 = np.random.choice(np.arange(0, w0 // 4))

        crop_size = min(random.randint(w0 // 4, w0 // 4 * 3), 128)  # < 400
        img_gt = img_gt[h1:h1 + crop_size // self.CheckNum, w1:w1 + crop_size]
        img_gt = cv2.resize(img_gt, (self.out_height * self.CheckNum, self.out_height), cv2.INTER_LINEAR)
        return img_gt

    def usm_sharp(self, img):
        """USM sharpening.
        input is 0~1 and BGR
        Input image: I; Blurry image: B.
        1. K = I + weight * (I - B)
        2. Mask = 1 if abs(I - B) > threshold, else: 0
        3. Blur mask:
        4. Out = Mask * K + (1 - Mask) * I
        """
        weight = np.random.randint(3, 7) / 10.0  # 0.5
        radius = np.random.randint(40, 60)  # 50
        threshold = np.random.randint(7, 15)  # 10
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        residual = img - blur
        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype('float32')
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

        K = img + weight * residual
        K = np.clip(K, 0, 1)
        simg = soft_mask * K + (1 - soft_mask) * img
        return simg

    def check_corpus(self, text):
        for i in text:
            if i not in self.CommonWords:
                return True
        if len(text) > self.CheckNum:
            return True
        return False

    def get_text(self):
        which_text = random.random() * 4
        if which_text > 1.8:
            text = self.corpus1[random.randint(0, len(self.corpus1) - 1)]
        elif which_text > 1.0:
            text = random.choices(self.CommonWords,
                                  k=random.randint(self.opt['min_text_length'], self.opt['max_text_length']))
            symble = random.choices(self.HalfWords, k=random.randint(2, 6))
            combine = text + symble
            random.shuffle(combine)
            text = "".join(combine)
            text = text[:int(self.opt['max_text_length'])]
            text = text.replace(' ', '').replace('\u3000', '')
        # else:#full half words
        #     symble = random.choices(self.HalfWords, k=random.randint(4, 16))
        #     text = "".join(symble)
        #     text = text[:int(self.opt['max_text_length'])]
        #     text = text.replace(' ', '').replace('\u3000', '')
        else:  # 3 english 4 number
            symble = random.choices(self.EnglishOnly, k=random.randint(3, 3)) + random.choices(self.NumberOnly,
                                                                                               k=random.randint(4, 4))
            text = "".join(symble)

        # else: #full half words
        #     text = random.choices(self.CommonWords, k=random.randint(self.opt['min_text_length'], self.opt['max_text_length']))
        #     text = "".join(text)
        #     text = text[:int(self.opt['max_text_length'])]
        #     text = text.replace(' ', '').replace('\u3000', '')
        #
        # # text = '你的账号有一些不正确的操'

        text = "".join(text.split())
        text = text.encode('unicode_escape').decode('utf-8').replace(' ', '')
        result = re.findall(r'\\x[a-f0-9]{2}', text)
        for x in result:
            text = text.replace(x, '')
        try:
            text = text.encode('utf-8').decode('unicode_escape')
        except:
            return [], []
        len_text = len(text)

        if len_text > self.CheckNum:
            x = len_text - self.CheckNum
            x0 = random.randint(0, x)
            y0 = x0 + random.randint(self.opt['min_text_length'], self.max_corpus_length)
            select_text = text[x0:int(min(y0, self.max_corpus_length + x0))]
        else:
            select_text = text

        if self.CheckNum == 1:
            ind = random.randint(0, len(self.CommonWords) - 1)
            select_text = self.CommonWords[ind]
            if len(select_text.replace('\u3000', '')) == 0:
                select_text = self.CommonWords[ind + random.randint(-100, 100)]
        # select_text = '请尽快以email的形式'
        # if random.random()>0.5:
        #     select_text = '黑NP2672'
        # else:
        #     select_text = '粤CPX357'
        if len(select_text) > 0:
            while len(select_text) < self.CheckNum:
                select_text += select_text
            select_text = select_text[:self.CheckNum]

        # print("select_text:", select_text)

        check_text = ''
        label = []
        for i in select_text:
            index = self.CommonWords.find(i)
            if index >= 0:
                check_text = check_text + i
                label.append(index)

        return check_text, label

    def __getitem__(self, index):
        for iii in range(1):
            # try:
            index = random.randint(0, len(self.paths) - 1)
            gt_path = self.paths[index]
            img_gt = self.get_valid_patch(gt_path)

            im_PIL = img_gt[:, :, ::-1] * 255  # to RGB
            im_PIL = Image.fromarray(im_PIL.astype(np.uint8))  # RGB 0~255 H*W*C

            text_img, mask_img, text, label_gt, char_locs = self.image_add_text(im_PIL)  # output BGR 0~1
            while np.sum(mask_img) < 1.0:  # remove these with null output
                text_img, mask_img, text, label_gt, char_locs = self.image_add_text(im_PIL)  # output BGR 0~1
            # which_seg = random.random()
            # if which_seg > 0:
            #     w_offset = random.choice([0,512])
            # else:
            #     w_offset = random.randint(0,512)
            # text_img = text_img[:, w_offset:w_offset+512,:]
            # mask_img = mask_img[:, w_offset:w_offset+512,:]

            # print([iii, text, char_locs])
            # print([iii])
            # color jitter
            brightness = self.opt.get('brightness', (0.9, 1.1))
            contrast = self.opt.get('contrast', (0.9, 1.1))
            saturation = self.opt.get('saturation', (0.9, 1.1))
            # hue = self.opt.get('hue', (-0.1, 0.1))
            hue = self.opt.get('hue', None)
            text_img = self.color_jitter_pt(img2tensor(text_img, bgr2rgb=True, float32=False), brightness, contrast,
                                            saturation, hue)  # RGB Tensor 0~1 C*H*W
            text_img = text_img.numpy().transpose(1, 2, 0)[:, :,
                       ::-1]  # transfer back to numpy for the following degradation, 0~1, BGR, H*W*C

            # if text_img.shape[-2] > 128*self.CheckNum:
            #     text_img = cv2.resize(text_img, (128*self.CheckNum, text_img.shape[-3]), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            try:
                degradation_type = random.random()
                if degradation_type > 0.6:  # real-esrgan
                    ##input should be BGR 0~1 numpy H*W*C
                    ##output is RGB 0~1 tensor
                    lq = real_esrgan_degradation(text_img, insf=2).squeeze(0).detach().numpy()  # output numpy c*h*w 0~1 RGB random.choice([2, 4]
                    lq = lq.transpose((1, 2, 0))  # transfer to h*w*c
                elif degradation_type > 0.05:
                    ##input should be RGB 0~1 numpy H*W*C
                    ##output is RGB 0~1 numpy H*W*C
                    gt_tmp = text_img[:, :, ::-1]  # transfer to RGB
                    lq, _ = bsrgan_degradation(gt_tmp, sf=2), #random.choice([4, 6]lq_patchsize=None) # RGB 0~1 numpy h*w*c
                    lq = lq.astype(np.float32)
                else:
                    lq = text_img[:, :, ::-1] # out RGB [:, :, ::-1]
                    pass
            except Exception as e:
                # gt_tmp = text_img[:,:,::-1]#transfer to RGB
                # lq, _ = bsrgan_degradation(gt_tmp, sf=random.choice([1,2]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
                # lq = lq.astype(np.float32)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(['error degradation', text_img.shape, e, exc_type, fname, exc_tb.tb_lineno])
                lq = np.ascontiguousarray(text_img[:, :, ::-1]) # BGR  # out RGB #[:, :, ::-1]
            # elif degradation_type > 0:
            #     gt_tmp = text_img[:,:,::-1]#transfer to RGB
            #     lq = self.AddDownSample(text_img)
            #     lq = lq.astype(np.float32)

            h_lq, w_lq = lq.shape[:2]
            # try:
            # print("scale:", 32 / h_lq)
            lq = cv2.resize(lq, (0, 0), fx=(self.out_height // self.scale) / h_lq, fy=(self.out_height // self.scale) / h_lq,
                            interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            # lq2 = cv2.resize(lq, (round(w_lq*32/h_lq), 32),  interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            # except:
            #     print(['check size', lq.shape, text_img.shape])
            #     exit('ssss')
            #
            '''
            TextGTFillBG = np.zeros((128, 128 * self.CheckNum, 3)).astype(text_img.dtype)
            MaskFillBG = np.zeros((128, 128 * self.CheckNum, 3))
            TextLQFillBG = np.zeros((32, 32 * self.CheckNum, 3)).astype(lq.dtype)
            if text_img.shape[-2] < 128 * self.CheckNum:
                TextGTFillBG[:, :text_img.shape[-2], :] = TextGTFillBG[:, :text_img.shape[-2], :] + text_img
                text_img = TextGTFillBG
                MaskFillBG[:, :mask_img.shape[-2], :] = MaskFillBG[:, :mask_img.shape[-2], :] + mask_img
                mask_img = MaskFillBG
            if lq.shape[-2] < 32 * self.CheckNum:
                TextLQFillBG[:, :lq.shape[-2], :] = TextLQFillBG[:, :lq.shape[-2], :] + lq
                lq = TextLQFillBG
            if lq.shape[-2] > 32 * self.CheckNum or text_img.shape[-2] > 128 * self.CheckNum:
                print(['error size', lq.shape, text_img.shape])

                # lq = cv2.resize(lq, (32*self.CheckNum, 32), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            '''
            # print("scale:", self.scale)
            TextGTFillBG = np.zeros((self.out_height, self.out_width, 3)).astype(text_img.dtype)
            MaskFillBG = np.zeros((self.out_height, self.out_width, 3))
            TextLQFillBG = np.zeros((self.out_height // self.scale, self.out_width // self.scale, 3)).astype(lq.dtype)
            if text_img.shape[-2] < self.out_height * self.CheckNum:
                TextGTFillBG[:, :text_img.shape[-2], :] = TextGTFillBG[:, :text_img.shape[-2], :] + text_img
                text_img = TextGTFillBG
                MaskFillBG[:, :mask_img.shape[-2], :] = MaskFillBG[:, :mask_img.shape[-2], :] + mask_img
                mask_img = MaskFillBG
            if lq.shape[-2] < self.out_height // self.scale * self.CheckNum:
                TextLQFillBG[:, :lq.shape[-2], :] = TextLQFillBG[:, :lq.shape[-2], :] + lq
                lq = TextLQFillBG
            if lq.shape[-2] > self.out_height // self.scale * self.CheckNum or text_img.shape[-2] > self.out_height * self.CheckNum:
                print(['error size', lq.shape, text_img.shape])

            if random.random() > 0.5:
                rescale = random.random() + 1.0
                H_hq, W_hq = TextGTFillBG.shape[:2]
                H_lq, W_lq = TextLQFillBG.shape[:2]
                TextGTFillBG = cv2.resize(TextGTFillBG, (int(W_hq * rescale), H_hq), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
                MaskFillBG = cv2.resize(MaskFillBG, (int(W_hq * rescale), H_hq), interpolation=random.choice(
                    [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
                TextLQFillBG = cv2.resize(TextLQFillBG, (int(W_lq * rescale), H_lq), interpolation=random.choice(
                    [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))

                text_img = TextGTFillBG[:, :W_hq]
                mask_img = MaskFillBG[:, :W_hq]
                lq = TextLQFillBG[:, :W_lq]

            if len(label_gt) < self.CheckNum:
                for i in range(len(text), self.CheckNum):
                    label_gt.append(6735)

            if index % 50 == 0:
                save_dir = "HQSynth_v2"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                # print("save:", np.unique(lq))
                cv2.imwrite(os.path.join(save_dir, str(index) + "_lq.jpg"), (lq * 255.0).astype(np.uint8))
                cv2.imwrite(os.path.join(save_dir, str(index) + "_hq.jpg"), (text_img[:, :, ::-1] * 255.0).astype(np.uint8))

            text_img = img2tensor(text_img, bgr2rgb=True, float32=False)  # RGB 0~1
            mask_img = img2tensor(mask_img, bgr2rgb=True, float32=False)  # RGB 0~1
            lq = img2tensor(lq, bgr2rgb=False, float32=False)  # RGB 0~1

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(text_img, self.mean, self.std, inplace=True)  # -1~1 RGB
                normalize(lq, self.mean, self.std, inplace=True)  # -1~1 RGB

        label = torch.Tensor(label_gt).type(torch.LongTensor)
        char_locs = torch.Tensor(char_locs) / (self.CheckNum * self.out_height)

        # print(lq.shape, text_img.shape, label)

        return {'lq': lq, 'gt': text_img, 'lq_path': 'None', 'gt_path': 'None', 'label': label}
        #return {'gt': text_img, 'mask': mask_img, 'label': label, 'lq': lq,
        #        'boxinfo': char_locs}  # 'gt_path': gt_path, 'mask':mask_img,


    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class TextDegradationDatasetwCanny(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(TextDegradationDatasetwCanny, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.dataroot = opt['path_bg']
        self.gt_folder = opt['path_bg']
        # self.mask_folder = opt['maskroot']
        self.mean = opt.get("mean", None)#['mean']
        self.std = opt.get("std", None)
        self.out_height = opt['gt_size'][0]
        self.out_width = opt['gt_size'][1]
        self.scale = opt['scale']
        self.CheckNum = int(opt['check_num'])

        # self.CommonWords = opt['CommonWords']
        self.CommonWords = alphabet

        index_not_chinese = [35, 107, 114, 165, 194, 206, 226, 227, 229, 248, 273, 302, 317, 333, 355, 371, 377, 398,
                             417, 434, 437, 456, 466, 468, 506, 536, 565, 575, 585, 595, 625, 634, 653, 661, 665, 667,
                             732, 756, 815, 841, 920, 935, 949, 957, 1020, 1024, 1028, 1064, 1067, 1133, 1158, 1221,
                             1230, 1258, 1341, 1343, 1369, 1417, 1457, 1496, 1497, 1499, 1517, 1547, 1564, 1600, 1624,
                             1637, 1662, 1693, 1770, 1780, 1802, 1835, 1855, 1857, 1925, 1947, 1958, 1959, 2016, 2031,
                             2034, 2035, 2037, 2059, 2072, 2083, 2089, 2116, 2173, 2217, 2237, 2315, 2330, 2346, 2368,
                             2404, 2435, 2436, 2446, 2451, 2456, 2485, 2488, 2499, 2556, 2557, 2559, 2582, 2583, 2607,
                             2616, 2679, 2680, 2707, 2735, 2748, 2771, 2799, 2865, 2883, 2895, 2913, 2933, 2963, 2991,
                             3063, 3074, 3087, 3110, 3126, 3128, 3129, 3157, 3177, 3178, 3185, 3212, 3213, 3230, 3242,
                             3253, 3258, 3278, 3292, 3303, 3305, 3319, 3367, 3370, 3411, 3437, 3478, 3484, 3486, 3570,
                             3583, 3591, 3612, 3671, 3675, 3708, 3762, 3782, 3790, 3815, 3817, 3836, 3867, 3872, 3906,
                             3913, 3914, 3924, 3955, 3960, 3975, 3993, 4013, 4035, 4041, 4053, 4063, 4068, 4114, 4123,
                             4124, 4166, 4179, 4190, 4216, 4312, 4324, 4333, 4368, 4378, 4431, 4443, 4449, 4458, 4467,
                             4498, 4537, 4552, 4566, 4623, 4637, 4654, 4658, 4668, 4738, 4753, 4784, 4932, 4975, 4979,
                             4983, 5005, 5016, 5026, 5033, 5036, 5051, 5052, 5076, 5082, 5090, 5129, 5135, 5166, 5175,
                             5206, 5243, 5252, 5274, 5288, 5319, 5355, 5370, 5411, 5455, 5468, 5497, 5504, 5654, 5670,
                             5688, 5702, 5734, 5741, 5749, 5812, 5815, 5829, 5842, 5850, 5870, 5880, 5888, 5921, 5955,
                             5957, 5965, 5967, 5994, 5995, 5996, 6021, 6036, 6064, 6071, 6104, 6119, 6166, 6190, 6218,
                             6237, 6251, 6254, 6280, 6286, 6311, 6329, 6344, 6408, 6416, 6445, 6449, 6462, 6472, 6475,
                             6506, 6546, 6567, 6651, 6678, 6723, 6731]
        index_not_chinese = [3074, 3914, 1959, 165, 6064, 3762, 5455, 5370, 4753, 6449, 2607, 4368, 6344, 1064, 2616,
                             1024, 1958, 841, 3278, 5870, 3177, 4449, 5888, 1637, 333, 2059, 5129, 2559, 302, 5076, 434,
                             5670, 2217, 6021, 6445, 2913, 5243, 3790, 2037, 665, 4333, 2034, 2404, 3906, 3671, 5036,
                             4053, 2679, 3486, 6071, 114, 3230]
        english_letter = [3074, 3914, 1959, 165, 6064, 3762, 5455, 5370, 4753, 6449, 2607, 4368, 6344, 1064, 2616, 1024,
                          1958, 841, 3278, 5870, 3177, 4449, 5888, 1637, 333, 2059]
        number_letter = [575, 2116, 1230, 1857, 3157, 1564, 4124, 3708, 2072, 355]
        self.HalfWords = []
        for ind in index_not_chinese:
            self.HalfWords.append(alphabet[ind])
        self.EnglishOnly = []
        self.NumberOnly = []
        for ind in english_letter:
            self.EnglishOnly.append(alphabet[ind])
        for ind in number_letter:
            self.NumberOnly.append(alphabet[ind])
        # self.NumWords = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#？'
        # self.SymbolWords = string.ascii_letters + string.digits + string.punctuation
        with open(opt['corpus_path1'], 'r') as f:
            lines = f.read().split('\n')
        self.corpus1 = [line.replace("\n", "").split("\t")[3].strip() for line in lines if len(line) > opt['min_text_length']]
        self.max_corpus_length = opt['max_text_length']

        # print("corpus1:", self.corpus1)

        print("[Corpus1] Number of Corpus1 pairs: {}, classes: {}".format(len(self.corpus1), len(self.CommonWords)))

        # self.gray_aug = ia.Grayscale(alpha=random.random())
        self.gray_aug = ia.Grayscale(alpha=random.randint(5, 10) / 10.0)
        error_flg = False
        # file client (lmdb io backend)
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(os.path.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        elif self.io_backend_opt['type'] == 'disk':
            self.paths = []
            if os.path.isfile(self.opt['path_bg']):
                fp = open(self.opt['path_bg'], "r")
                lines = fp.read().split("\n")
                lines = [line.strip() for line in lines if len(line)]

                for line in lines:
                    image_file = os.path.join(self.dataroot, line)
                    if not os.path.exists(image_file):
                        print("%s does not exist!" % (image_file))
                        error_flg = True
                    self.paths.append(image_file)
            else:
                self.paths = paths_from_folder(self.gt_folder)

            if error_flg:
                raise ValueError("Some file paths are corrupted! Please re-check your file paths!")
            index = np.arange(len(self.paths))
            np.random.shuffle(index)
            self.paths = np.array(self.paths)
            self.paths = self.paths[index]
            print("[Dataset] Number of Bcakground pairs:", len(self.paths))

            self.font_paths = []
            # Fonts = ['Deng.ttf','Dengb.ttf','Dengl.ttf','simfang.ttf','simhei.ttf','simkai.ttf','simsun.ttc','STSONG.TTF']
            Fonts = os.listdir(opt['path_font'])
            for f in Fonts:
                self.font_paths.append(os.path.join(opt['path_font'], f))

        else:
            # disk backend: scan file list from a folder
            self.paths = paths_from_folder(self.gt_folder)
        self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

    def make_dataset(self, dirs):
        images = []
        assert os.path.isdir(dirs), '%s is not a valid directory' % dirs
        for root, _, fnames in sorted(os.walk(dirs)):
            fnames.sort()
            for fname in fnames:
                path = os.path.join(root, fname)
                images.append(path)
        return images

    def generate_canny_map(self, img):

        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    @staticmethod
    def color_jitter(img, shift):
        """jitter color: randomly jitter the RGB values, in numpy formats"""
        jitter_val = np.random.uniform(-shift, shift, 3).astype(np.float32)
        img = img + jitter_val
        img = np.clip(img, 0, 1)
        return img

    @staticmethod
    def color_jitter_pt(img, brightness, contrast, saturation, hue):
        """jitter color: randomly jitter the brightness, contrast, saturation, and hue, in torch Tensor formats"""
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and brightness is not None:
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = adjust_brightness(img, brightness_factor)

            if fn_id == 1 and contrast is not None:
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = adjust_contrast(img, contrast_factor)

            if fn_id == 2 and saturation is not None:
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = adjust_saturation(img, saturation_factor)

            if fn_id == 3 and hue is not None:
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = adjust_hue(img, hue_factor)
        return img

    def AddDownSample(self, img):  # downsampling
        sampler = random.randint(15, 35) * 1.0
        h0, w0 = img.shape[:2]
        if random.random() > 0.5:
            img = cv2.resize(img, (int(w0 / sampler * 10.0), int(h0 / sampler * 10.0)), cv2.INTER_LINEAR)
        else:
            img = cv2.resize(img, (int(w0 / sampler * 10.0), int(h0 / sampler * 10.0)), cv2.INTER_LINEAR)
        out = cv2.resize(img.copy(), (self.out_size // 2, self.out_size // 2), cv2.INTER_LINEAR)
        return out

    def image_add_text(self, img):
        '''
        Get the text
        '''
        text, label = self.get_text()
        while self.check_corpus(text):
            text, label = self.get_text()

        w, h = img.size
        bg_is_white = 0
        if random.random() > 0.5:  # 0.96: #white bg
            img_tmp = Image.new('RGB', (w, h), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            # img_tmp = Image.new('RGB', (w, h), (255,255,255))
            im_numpy = np.array(img_tmp)[:, :, ::-1]  # BGR 0~255
            img_aug = self.gray_aug(image=im_numpy.astype(np.uint8))
            img = Image.fromarray(img_aug[:, :, ::-1])
            img = img_tmp
            bg_is_white = 1

        # 字体的格式 这里的SimHei.ttf需要有这个字体
        fontpath = self.font_paths[random.randint(0, len(self.font_paths) - 1)]
        text_size = random.randint(32, 48)  # if Small > 0.3 else random.randint(128, 235)#235 267 (80, 140)
        x = random.randint(-10, 30)  #
        y = random.randint(-10, 10)  #
        pos = (x, y)  # width, height from top left to bottom right
        fontStyle = ImageFont.truetype(fontpath, text_size, encoding="utf-8")
        mask_color = (255, 255, 255)

        pos_mask = Image.new('L', (w, h), 0)
        drawer_tmp = ImageDraw.Draw(pos_mask)
        char_locs = []
        img_max_width = 0

        for i in range(1, len(text) + 1):
            p = text[:i]
            drawer_tmp.text(pos, p, font=fontStyle, fill=255)
            char_mask = np.array(pos_mask).copy()
            vertical_projection = np.sum(char_mask, axis=0)
            ws = np.where(vertical_projection > 1)  # height, width
            locs = list(ws[0])
            if len(locs) == 0:  # space
                continue
            # if i == 1:
            if len(char_locs) == 0:
                char_locs.append(max(min(locs), 0))
                char_locs.append(min(max(locs), w - 1))
            else:
                new_locs = []
                for l in locs:
                    if l > char_locs[-1]:
                        new_locs.append(l)
                if len(new_locs):
                    char_locs.append(max(min(new_locs), 0))
                    char_locs.append(min(max(new_locs), w - 1))

        if len(char_locs) == 0:
            # print(['error max char_locs', text])
            return None, np.zeros((1, 1)), None, None, None
        img_max_width = max(char_locs)
        if len(text) != len(char_locs) // 2 or len(label) != len(char_locs) // 2 or img_max_width > self.out_height * self.CheckNum:
            # print(['return error inconsistent numbers', text, len(text), len(label), len(char_locs)])
            return None, np.zeros((1, 1)), None, None, None

        if self.CheckNum == 1:
            char_locs = [0, self.out_height]

        for i in range(len(text), self.CheckNum):
            char_locs.append(self.CheckNum * self.out_height)
            char_locs.append(self.CheckNum * self.out_height)

        which_color = random.random()  # common
        if bg_is_white:
            if random.random() > 0.9:
                text_color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))
            else:
                text_color = (0, 0, 0)
        else:
            text_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        text_color = (0, 0, 0)
        # if 1: # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #     pos = (0,0)
        #     text_size = 128
        #     fontpath = self.font_paths[0]
        #     img = Image.new('RGB', (w, h), (255,255,255))
        #     text_color = (0,0,0)

        drawer = ImageDraw.Draw(img)
        drawer.text(pos, text, font=fontStyle, fill=text_color)
        # drawer_mask.text(pos, text, font=fontStyle, fill=mask_color)
        pos_mask = np.array(pos_mask)
        mask = np.repeat(pos_mask[:, :, np.newaxis], 3, axis=2)
        mask[mask > 128] = 255
        mask[mask <= 128] = 0

        img = np.array(img)[:, :, ::-1].astype(np.float32)
        offset_w = min(img_max_width + random.randint(0, 16), 128 * self.CheckNum)
        offset_w = offset_w // 4 * 4

        img = img[:, :offset_w, :]
        mask = mask[:, :offset_w, :]

        if img.shape[-2] < 10 or img.shape[-3] < 10 or img.shape[-2] > 128 * self.CheckNum:
            print('return error 1')
            return None, np.zeros((1, 1)), None, None, None
        return img / 255.0, mask / 255.0, text, label, char_locs

    def read_valid_image(self, gt_path):
        retry = 3
        while retry > 0:
            try:
                # img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)/255.0
                img_bytes = self.file_client.get(gt_path)
                img_gt = imfrombytes(img_bytes, float32=True)
            except (IOError, OSError) as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                gt_path = self.paths[random.randint(0, len(self.paths) - 1)]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        return img_gt

    def get_valid_patch(self, gt_path):
        img_gt = self.read_valid_image(gt_path)
        ##flip
        flip = 1 if random.random() > 0.5 else 0
        if flip:
            cv2.flip(img_gt, 1, img_gt)
        random_size = random.randint(320, 400)
        img_gt = cv2.resize(img_gt, (random_size, random_size), cv2.INTER_LINEAR)
        h0, w0 = img_gt.shape[:2]

        h1 = np.random.choice(np.arange(0, h0 // 2))
        w1 = np.random.choice(np.arange(0, w0 // 4))

        crop_size = min(random.randint(w0 // 4, w0 // 4 * 3), 128)  # < 400
        img_gt = img_gt[h1:h1 + crop_size // self.CheckNum, w1:w1 + crop_size]
        img_gt = cv2.resize(img_gt, (self.out_height * self.CheckNum, self.out_height), cv2.INTER_LINEAR)
        return img_gt

    def usm_sharp(self, img):
        """USM sharpening.
        input is 0~1 and BGR
        Input image: I; Blurry image: B.
        1. K = I + weight * (I - B)
        2. Mask = 1 if abs(I - B) > threshold, else: 0
        3. Blur mask:
        4. Out = Mask * K + (1 - Mask) * I
        """
        weight = np.random.randint(3, 7) / 10.0  # 0.5
        radius = np.random.randint(40, 60)  # 50
        threshold = np.random.randint(7, 15)  # 10
        if radius % 2 == 0:
            radius += 1
        blur = cv2.GaussianBlur(img, (radius, radius), 0)
        residual = img - blur
        mask = np.abs(residual) * 255 > threshold
        mask = mask.astype('float32')
        soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

        K = img + weight * residual
        K = np.clip(K, 0, 1)
        simg = soft_mask * K + (1 - soft_mask) * img
        return simg

    def check_corpus(self, text):
        for i in text:
            if i not in self.CommonWords:
                return True
        if len(text) > self.CheckNum:
            return True
        return False

    def get_text(self):
        which_text = random.random() * 4
        if which_text > 1.8:
            text = self.corpus1[random.randint(0, len(self.corpus1) - 1)]
        elif which_text > 1.0:
            text = random.choices(self.CommonWords,
                                  k=random.randint(self.opt['min_text_length'], self.opt['max_text_length']))
            symble = random.choices(self.HalfWords, k=random.randint(2, 6))
            combine = text + symble
            random.shuffle(combine)
            text = "".join(combine)
            text = text[:int(self.opt['max_text_length'])]
            text = text.replace(' ', '').replace('\u3000', '')
        # else:#full half words
        #     symble = random.choices(self.HalfWords, k=random.randint(4, 16))
        #     text = "".join(symble)
        #     text = text[:int(self.opt['max_text_length'])]
        #     text = text.replace(' ', '').replace('\u3000', '')
        else:  # 3 english 4 number
            symble = random.choices(self.EnglishOnly, k=random.randint(3, 3)) + random.choices(self.NumberOnly,
                                                                                               k=random.randint(4, 4))
            text = "".join(symble)

        # else: #full half words
        #     text = random.choices(self.CommonWords, k=random.randint(self.opt['min_text_length'], self.opt['max_text_length']))
        #     text = "".join(text)
        #     text = text[:int(self.opt['max_text_length'])]
        #     text = text.replace(' ', '').replace('\u3000', '')
        #
        # # text = '你的账号有一些不正确的操'

        text = "".join(text.split())
        text = text.encode('unicode_escape').decode('utf-8').replace(' ', '')
        result = re.findall(r'\\x[a-f0-9]{2}', text)
        for x in result:
            text = text.replace(x, '')
        try:
            text = text.encode('utf-8').decode('unicode_escape')
        except:
            return [], []
        len_text = len(text)

        if len_text > self.CheckNum:
            x = len_text - self.CheckNum
            x0 = random.randint(0, x)
            y0 = x0 + random.randint(self.opt['min_text_length'], self.max_corpus_length)
            select_text = text[x0:int(min(y0, self.max_corpus_length + x0))]
        else:
            select_text = text

        if self.CheckNum == 1:
            ind = random.randint(0, len(self.CommonWords) - 1)
            select_text = self.CommonWords[ind]
            if len(select_text.replace('\u3000', '')) == 0:
                select_text = self.CommonWords[ind + random.randint(-100, 100)]
        # select_text = '请尽快以email的形式'
        # if random.random()>0.5:
        #     select_text = '黑NP2672'
        # else:
        #     select_text = '粤CPX357'
        if len(select_text) > 0:
            while len(select_text) < self.CheckNum:
                select_text += select_text
            select_text = select_text[:self.CheckNum]

        # print("select_text:", select_text)

        check_text = ''
        label = []
        for i in select_text:
            index = self.CommonWords.find(i)
            if index >= 0:
                check_text = check_text + i
                label.append(index)

        return check_text, label

    def __getitem__(self, index):
        for iii in range(1):
            # try:
            index = random.randint(0, len(self.paths) - 1)
            gt_path = self.paths[index]
            img_gt = self.get_valid_patch(gt_path)

            im_PIL = img_gt[:, :, ::-1] * 255  # to RGB
            im_PIL = Image.fromarray(im_PIL.astype(np.uint8))  # RGB 0~255 H*W*C

            text_img, mask_img, text, label_gt, char_locs = self.image_add_text(im_PIL)  # output BGR 0~1
            while np.sum(mask_img) < 1.0:  # remove these with null output
                text_img, mask_img, text, label_gt, char_locs = self.image_add_text(im_PIL)  # output BGR 0~1
            # which_seg = random.random()
            # if which_seg > 0:
            #     w_offset = random.choice([0,512])
            # else:
            #     w_offset = random.randint(0,512)
            # text_img = text_img[:, w_offset:w_offset+512,:]
            # mask_img = mask_img[:, w_offset:w_offset+512,:]

            # print([iii, text, char_locs])
            # print([iii])
            # color jitter
            brightness = self.opt.get('brightness', (0.9, 1.1))
            contrast = self.opt.get('contrast', (0.9, 1.1))
            saturation = self.opt.get('saturation', (0.9, 1.1))
            # hue = self.opt.get('hue', (-0.1, 0.1))
            hue = self.opt.get('hue', None)
            text_img = self.color_jitter_pt(img2tensor(text_img, bgr2rgb=True, float32=False), brightness, contrast,
                                            saturation, hue)  # RGB Tensor 0~1 C*H*W
            text_img = text_img.numpy().transpose(1, 2, 0)[:, :,
                       ::-1]  # transfer back to numpy for the following degradation, 0~1, BGR, H*W*C

            # if text_img.shape[-2] > 128*self.CheckNum:
            #     text_img = cv2.resize(text_img, (128*self.CheckNum, text_img.shape[-3]), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            try:
                degradation_type = random.random()
                if degradation_type > 0.6:  # real-esrgan
                    ##input should be BGR 0~1 numpy H*W*C
                    ##output is RGB 0~1 tensor
                    lq = real_esrgan_degradation(text_img, insf=2).squeeze(0).detach().numpy()  # output numpy c*h*w 0~1 RGB random.choice([2, 4]
                    lq = lq.transpose((1, 2, 0))  # transfer to h*w*c
                elif degradation_type > 0.05:
                    ##input should be RGB 0~1 numpy H*W*C
                    ##output is RGB 0~1 numpy H*W*C
                    gt_tmp = text_img[:, :, ::-1]  # transfer to RGB
                    lq, _ = bsrgan_degradation(gt_tmp, sf=2), #random.choice([4, 6]lq_patchsize=None) # RGB 0~1 numpy h*w*c
                    lq = lq.astype(np.float32)
                else:
                    lq = text_img[:, :, ::-1] # out RGB [:, :, ::-1]
                    pass
            except Exception as e:
                # gt_tmp = text_img[:,:,::-1]#transfer to RGB
                # lq, _ = bsrgan_degradation(gt_tmp, sf=random.choice([1,2]), lq_patchsize=None)#RGB 0~1 numpy h*w*c
                # lq = lq.astype(np.float32)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                # print(['error degradation', text_img.shape, e, exc_type, fname, exc_tb.tb_lineno])
                lq = np.ascontiguousarray(text_img[:, :, ::-1]) # BGR  # out RGB #[:, :, ::-1]
            # elif degradation_type > 0:
            #     gt_tmp = text_img[:,:,::-1]#transfer to RGB
            #     lq = self.AddDownSample(text_img)
            #     lq = lq.astype(np.float32)

            h_lq, w_lq = lq.shape[:2]
            # try:
            # print("scale:", 32 / h_lq)
            lq = cv2.resize(lq, (0, 0), fx=(self.out_height // self.scale) / h_lq, fy=(self.out_height // self.scale) / h_lq,
                            interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            # lq2 = cv2.resize(lq, (round(w_lq*32/h_lq), 32),  interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            # except:
            #     print(['check size', lq.shape, text_img.shape])
            #     exit('ssss')
            #
            '''
            TextGTFillBG = np.zeros((128, 128 * self.CheckNum, 3)).astype(text_img.dtype)
            MaskFillBG = np.zeros((128, 128 * self.CheckNum, 3))
            TextLQFillBG = np.zeros((32, 32 * self.CheckNum, 3)).astype(lq.dtype)
            if text_img.shape[-2] < 128 * self.CheckNum:
                TextGTFillBG[:, :text_img.shape[-2], :] = TextGTFillBG[:, :text_img.shape[-2], :] + text_img
                text_img = TextGTFillBG
                MaskFillBG[:, :mask_img.shape[-2], :] = MaskFillBG[:, :mask_img.shape[-2], :] + mask_img
                mask_img = MaskFillBG
            if lq.shape[-2] < 32 * self.CheckNum:
                TextLQFillBG[:, :lq.shape[-2], :] = TextLQFillBG[:, :lq.shape[-2], :] + lq
                lq = TextLQFillBG
            if lq.shape[-2] > 32 * self.CheckNum or text_img.shape[-2] > 128 * self.CheckNum:
                print(['error size', lq.shape, text_img.shape])

                # lq = cv2.resize(lq, (32*self.CheckNum, 32), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
            '''
            # print("scale:", self.scale)
            TextGTFillBG = np.zeros((self.out_height, self.out_width, 3)).astype(text_img.dtype)
            MaskFillBG = np.zeros((self.out_height, self.out_width, 3))
            TextLQFillBG = np.zeros((self.out_height // self.scale, self.out_width // self.scale, 3)).astype(lq.dtype)
            if text_img.shape[-2] < self.out_height * self.CheckNum:
                TextGTFillBG[:, :text_img.shape[-2], :] = TextGTFillBG[:, :text_img.shape[-2], :] + text_img
                text_img = TextGTFillBG
                MaskFillBG[:, :mask_img.shape[-2], :] = MaskFillBG[:, :mask_img.shape[-2], :] + mask_img
                mask_img = MaskFillBG
            if lq.shape[-2] < self.out_height // self.scale * self.CheckNum:
                TextLQFillBG[:, :lq.shape[-2], :] = TextLQFillBG[:, :lq.shape[-2], :] + lq
                lq = TextLQFillBG
            if lq.shape[-2] > self.out_height // self.scale * self.CheckNum or text_img.shape[-2] > self.out_height * self.CheckNum:
                print(['error size', lq.shape, text_img.shape])

            if random.random() > 0.5:
                scale = random.random() * 0.5 + 1.
                H_hq, W_hq = TextGTFillBG.shape[:2]
                H_lq, W_lq = TextLQFillBG.shape[:2]
                TextGTFillBG = cv2.resize(TextGTFillBG, (int(W_hq * scale), H_hq), interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
                MaskFillBG = cv2.resize(MaskFillBG, (int(W_hq * scale), H_hq), interpolation=random.choice(
                    [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
                TextLQFillBG = cv2.resize(TextLQFillBG, (int(W_lq * scale), H_lq), interpolation=random.choice(
                    [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))

                text_img = TextGTFillBG[:, :W_hq]
                mask_img = MaskFillBG[:, :W_hq]
                lq = TextLQFillBG[:, :W_lq]

            if len(label_gt) < self.CheckNum:
                for i in range(len(text), self.CheckNum):
                    label_gt.append(6735)

            if index % 50 == 0:
                save_dir = "HQSynth_Canny"
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)
                # print("save:", np.unique(lq))
                cv2.imwrite(os.path.join(save_dir, str(index) + "_lq.jpg"), (lq * 255.0).astype(np.uint8))
                cv2.imwrite(os.path.join(save_dir, str(index) + "_hq.jpg"), (text_img[..., ::-1] * 255.0).astype(np.uint8))

            # Calculate the canny maps
            lr_canny_map = self.generate_canny_map(lq)
            hr_canny_map = self.generate_canny_map(text_img)
            lr_canny_map = torch.tensor(lr_canny_map).unsqueeze(0)
            hr_canny_map = torch.tensor(hr_canny_map).unsqueeze(0)

            text_img = img2tensor(text_img, bgr2rgb=True, float32=False)  # RGB 0~1
            mask_img = img2tensor(mask_img, bgr2rgb=True, float32=False)  # RGB 0~1
            lq = img2tensor(lq, bgr2rgb=False, float32=False)  # RGB 0~1

            # print("LR data shapes:", lq.shape, lr_canny_map.shape)
            # print("HR data shapes:", text_img.shape, hr_canny_map.shape)

            # normalize
            if self.mean is not None or self.std is not None:
                normalize(text_img, self.mean, self.std, inplace=True)  # -1~1 RGB
                normalize(lq, self.mean, self.std, inplace=True)  # -1~1 RGB

            # Concatenate the maps
            lq = torch.cat([lq, lr_canny_map], dim=0)
            text_img = torch.cat([text_img, hr_canny_map], dim=0)

        label = torch.Tensor(label_gt).type(torch.LongTensor)
        char_locs = torch.Tensor(char_locs) / (self.CheckNum * self.out_height)

        # print(lq.shape, text_img.shape, label)

        return {'lq': lq, 'gt': text_img, 'lq_path': 'None', 'gt_path': 'None', 'label': label}

    def __len__(self):
        return len(self.paths)


import pickle
@DATASET_REGISTRY.register()
class PairedImageDatasetTextLine(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetTextLine, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_size = [64, 256]

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = []
        for folder in self.gt_folder:
            if "SynthBG" in folder:
                if 'filename_tmpl' in opt:
                    self.filename_tmpl = opt['filename_tmpl']
                else:
                    self.filename_tmpl = '{}'

                # if self.io_backend_opt['type'] == 'pickle':
                flist = os.listdir(folder)
                self.root_dir = folder
                self.database_dict = {}
                print("Loading pkl files from", folder, "...")
                for f in flist:
                    if f.endswith(".pkl"):
                        print("f:", f)
                        with open(os.path.join(folder, f), "rb") as pkl_f:
                            self.database_dict.update(pickle.load(pkl_f))
                self.keys = list(self.database_dict.keys())
                for imkey in self.keys:
                    HR_path = os.path.join(self.root_dir, imkey + ".jpg")
                    self.paths.append({"lq_path": HR_path, "gt_path": HR_path, 'syn_degr': False, "gt_annos": []})

                print("nSamples SynthBG:", len(self.keys))

            elif "CSVTR" in folder:

                self.imdir = os.path.join(folder, "filter_dir")
                self.gt_file = os.path.join(folder, "filter_train_" + ("train" if self.opt['phase'] == 'train' else "test") + ".list")

                self.gt_pairs = []

                gt_lines = open(self.gt_file, "r").readlines()
                for line in gt_lines:
                    items = line.replace("\n", "").split("\t")
                    # self.gt_pairs.append([os.path.join(self.imdir, items[2]), items[3]])
                    HR_path = os.path.join(self.imdir, items[2])
                    self.paths.append({"lq_path": HR_path, "gt_path": HR_path, 'syn_degr': False, "gt_annos": [items[3]]})

                print("nSamples CSVTR:", len(gt_lines))

        self.nSamples = len(self.paths)


    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        if random.random() > 0.6 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.6 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.6 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_lq

    def masking(self, image_hr, training=False):

        # masking half of the region

        H, W = image_hr.shape[:2]
        # print(H, W)
        W_m = int((random.random() * 0.3 + 0.3) * W)
        H_m = int((random.random() * 0.3 + 0.3) * H)

        x_m = int((W - W_m) * random.random())
        y_m = int((H - H_m) * random.random())

        image_lr = image_hr.copy()

        image_lr[y_m:y_m+H_m, x_m:x_m+W_m] = 0.0

        return image_lr

    def masking_v2(self, image_hr, training=False):

        # masking half of the region
        H_block, W_block = 4, 4

        H, W, C = image_hr.shape

        H_num = math.floor(H / H_block)
        W_num = math.floor(W / W_block)

        all_num = H_num * W_num

        masking = (np.random.rand(H_num, W_num) > 0.4).astype(np.uint8)[..., None]

        image_shuffle = image_hr.reshape(H // H_block, H_block, W // W_block, W_block, -1)
        image_shuffle = image_shuffle.transpose(0, 2, 1, 3, 4).reshape(H // H_block, W // W_block, -1)

        image_shuffle *= masking
        image_shuffle = image_shuffle.reshape(H // H_block, W // W_block, H_block, W_block, C).transpose(0, 2, 1, 3, 4)
        image_shuffle = image_shuffle.reshape(H, W, C)

        return image_shuffle

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_gt = cv2.resize(img_gt, (self.gt_size[1], self.gt_size[0]), interpolation=cv2.INTER_CUBIC)

        # print("img_lq:", img_lq.shape)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # img_lq = self.degradation(img_lq, self.opt['phase'] == "train")
            img_lq *= 255
            img_lq = degradation_pipeline(img_lq.astype(np.uint8))
            img_lq = img_lq.astype(np.float32) / 255.0
        # img_lq = self.masking_v2(img_lq)

        # Downsample
        H, W = img_gt.shape[:2]
        # print("img_hr:", H, W)
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        
        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetTextLinewCanny(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetTextLinewCanny, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_size = [64, 256]

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = []
        for folder in self.gt_folder:
            if "SynthBG" in folder:
                if 'filename_tmpl' in opt:
                    self.filename_tmpl = opt['filename_tmpl']
                else:
                    self.filename_tmpl = '{}'

                # if self.io_backend_opt['type'] == 'pickle':
                flist = os.listdir(folder)
                self.root_dir = folder
                self.database_dict = {}
                print("Loading pkl files from", folder, "...")
                for f in flist:
                    if f.endswith(".pkl"):
                        print("f:", f)
                        with open(os.path.join(folder, f), "rb") as pkl_f:
                            self.database_dict.update(pickle.load(pkl_f))
                self.keys = list(self.database_dict.keys())
                for imkey in self.keys:
                    HR_path = os.path.join(self.root_dir, imkey + ".jpg")
                    self.paths.append({"lq_path": HR_path, "gt_path": HR_path, 'syn_degr': False, "gt_annos": []})

                print("nSamples SynthBG:", len(self.keys))

            elif "CSVTR" in folder:

                self.imdir = os.path.join(folder, "filter_dir")
                self.gt_file = os.path.join(folder, "filter_train_" + (
                    "train" if self.opt['phase'] == 'train' else "test") + ".list")

                self.gt_pairs = []

                gt_lines = open(self.gt_file, "r").readlines()
                for line in gt_lines:
                    items = line.replace("\n", "").split("\t")
                    # self.gt_pairs.append([os.path.join(self.imdir, items[2]), items[3]])
                    HR_path = os.path.join(self.imdir, items[2])
                    self.paths.append(
                        {"lq_path": HR_path, "gt_path": HR_path, 'syn_degr': False, "gt_annos": [items[3]]})

                print("nSamples CSVTR:", len(gt_lines))

        self.nSamples = len(self.paths)

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        if random.random() > 0.6 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.6 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.6 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_lq

    def generate_canny_map(self, img):

        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    def masking(self, image_hr, training=False):

        # masking half of the region

        H, W = image_hr.shape[:2]
        # print(H, W)
        W_m = int((random.random() * 0.3 + 0.3) * W)
        H_m = int((random.random() * 0.3 + 0.3) * H)

        x_m = int((W - W_m) * random.random())
        y_m = int((H - H_m) * random.random())

        image_lr = image_hr.copy()

        image_lr[y_m:y_m + H_m, x_m:x_m + W_m] = 0.0

        return image_lr

    def masking_v2(self, image_hr, training=False):

        # masking half of the region
        H_block, W_block = 4, 4

        H, W, C = image_hr.shape

        H_num = math.floor(H / H_block)
        W_num = math.floor(W / W_block)

        all_num = H_num * W_num

        masking = (np.random.rand(H_num, W_num) > 0.4).astype(np.uint8)[..., None]

        image_shuffle = image_hr.reshape(H // H_block, H_block, W // W_block, W_block, -1)
        image_shuffle = image_shuffle.transpose(0, 2, 1, 3, 4).reshape(H // H_block, W // W_block, -1)

        image_shuffle *= masking
        image_shuffle = image_shuffle.reshape(H // H_block, W // W_block, H_block, W_block, C).transpose(0, 2, 1, 3, 4)
        image_shuffle = image_shuffle.reshape(H, W, C)

        return image_shuffle

    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        img_gt = cv2.resize(img_gt, (self.gt_size[1], self.gt_size[0]), interpolation=cv2.INTER_CUBIC)

        # print("img_lq:", img_lq.shape)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            # img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # img_lq = self.degradation(img_lq, self.opt['phase'] == "train")
            img_lq *= 255
            img_lq = degradation_pipeline(img_lq.astype(np.uint8))
            img_lq = img_lq.astype(np.float32) / 255.0
        # img_lq = self.masking_v2(img_lq)

        # Downsample
        H, W = img_gt.shape[:2]
        # print("img_hr:", H, W)
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)

        # Calculate the canny maps
        lr_canny_map = self.generate_canny_map(img_lq)
        hr_canny_map = self.generate_canny_map(img_gt)
        lr_canny_map = torch.tensor(lr_canny_map).unsqueeze(0)
        hr_canny_map = torch.tensor(hr_canny_map).unsqueeze(0)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]


        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Concatenate the maps
        img_lq = torch.cat([img_lq, lr_canny_map], dim=0)
        img_gt = torch.cat([img_gt, hr_canny_map], dim=0)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


def torch_rotate_img(torch_image_batches, arc_batches, rand_offs, off_range=0.2, mode='bicubic'):

    # ratios: H / W

    device = torch_image_batches.device

    N, C, H, W = torch_image_batches.shape
    ratios = H / float(W)

    # rand_offs = random.random() * (1 - ratios)
    ratios_mul = ratios + (rand_offs.unsqueeze(1) * off_range * 2) - off_range


    a11, a12, a21, a22 = torch.cos(arc_batches), \
                                     torch.sin(arc_batches), \
                                     -torch.sin(arc_batches), \
                                     torch.cos(arc_batches)

    # print("rand_offs:", rand_offs.shape, a12.shape)

    x_shift = torch.zeros_like(arc_batches)
    y_shift = torch.zeros_like(arc_batches)

    # print("device:", device)
    affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1) * ratios_mul, x_shift.unsqueeze(1),
                               a21.unsqueeze(1) / ratios_mul, a22.unsqueeze(1), y_shift.unsqueeze(1)], dim=1)
    affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

    affine_grid = F.affine_grid(affine_matrix, torch_image_batches.shape)
    distorted_batches = F.grid_sample(torch_image_batches, affine_grid, mode)

    return distorted_batches


def deformation(images, masks, scale_factor=1.0, rotate_train=5.0):

    # H, W, C = images[0].shape

    # Down-sample
    images = [cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)), interpolation=cv2.INTER_CUBIC) for image in images]
    masks = [torch.nn.functional.interpolate(mask.unsqueeze(0), (int(mask.shape[1] * scale_factor), int(mask.shape[2] * scale_factor)), mode='nearest').squeeze(0) for mask in masks]

    return images, masks


def deformation_v2(images, masks, scale_factor=1.0, rotate_train=10.0):

    # H, W, C = images[0].shape

    # Down-sample
    images = [cv2.resize(image, (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)), interpolation=cv2.INTER_CUBIC) for image in images]
    masks = [torch.nn.functional.interpolate(mask.unsqueeze(0), (int(mask.shape[1] * scale_factor), int(mask.shape[2] * scale_factor)), mode='nearest').squeeze(0) for mask in masks]

    batch_size = 1
    angle_batch = np.random.rand(batch_size) * rotate_train * 2 - rotate_train
    arc = angle_batch / 180. * math.pi
    rand_offs = torch.tensor(np.random.rand(batch_size)).float()

    arc = torch.tensor(arc).float()
    # print("1111")
    images_torch = []
    for cv_im in images:
        torch_im = torch.tensor(cv_im).float().permute(2, 0, 1).unsqueeze(0)
        torch_im = torch_rotate_img(torch_im, arc, rand_offs, mode='bicubic')
        images_torch.append(torch_im.permute(0, 2, 3, 1).squeeze(0).data.numpy())
    images = images_torch

    masks_torch = []
    for torch_mask in masks:
        torch_im = torch_mask.unsqueeze(0)
        torch_im = torch_rotate_img(torch_im, arc, rand_offs, mode='nearest')
        masks_torch.append(torch_im.squeeze(0))
    masks = masks_torch
    # print("2222")
    return images, masks


def get_TextZoom(opt, gt_folder):
    phase = opt['phase']#"val" #

    paths = []

    if phase == "train":
        gt_folder_1 = os.path.join(gt_folder, phase + "1_image") #"opt['phase']"
        gt_folder_2 = os.path.join(gt_folder, phase + "2_image")

        for idx in range(14573):
            hr_path = os.path.join(gt_folder_1, ("_image_hr-%09d" % (idx+1)) + ".jpg")
            lr_path = os.path.join(gt_folder_1, ("_image_lr-%09d" % (idx+1)) + ".jpg")

            paths.append(
            {"lq_path": lr_path, "gt_path": hr_path, 'syn_degr': False, "gt_annos": ["0,0,128,0,128,32,0,32,None"]})

        for idx in range(2794):
            hr_path = os.path.join(gt_folder_2, ("_image_hr-%09d" % (idx+1)) + ".jpg")
            lr_path = os.path.join(gt_folder_2, ("_image_lr-%09d" % (idx+1)) + ".jpg")

            paths.append(
            {"lq_path": lr_path, "gt_path": hr_path, 'syn_degr': False, "gt_annos": ["0,0,128,0,128,32,0,32,None"]})
    else:
        raise Exception("Sorry, TextZoom has no this scale", scale)
    return paths


def get_OtherDevice(opt, gt_folder):
    phase = opt['phase']#"val" #

    paths = []

    gt_folder = os.path.join(gt_folder) #"opt['phase']"

    imlist = os.listdir(gt_folder)

    for imname in imlist:

        impath = os.path.join(gt_folder, imname)

        paths.append(
        {"lq_path": impath, "gt_path": impath, 'syn_degr': False, "gt_annos": ["0,0,128,0,128,32,0,32,None"]})

    return paths


def get_RealCE(opt, gt_folder):
    phase = opt['phase']#"val" #
    gt_folder = os.path.join(gt_folder, phase) #"opt['phase']"

    fl_13mmdir = os.path.join(gt_folder, "13mm")
    fl_26mmdir = os.path.join(gt_folder, "26mm")
    fl_52mmdir = os.path.join(gt_folder, "52mm")

    paths = []

    scale = opt['scale']

    imlist = os.listdir(fl_13mmdir)
    if phase in ["val", "train"]:
        valid_list = open(os.path.join(gt_folder, "valid_list.txt"), "r").readlines()
        valid_list = [line.replace("\n", "") for line in valid_list]
        imlist = valid_list

    for imname in imlist:
        impath_13mm = os.path.join(fl_13mmdir, imname)
        impath_26mm = os.path.join(fl_26mmdir, imname)
        impath_52mm = os.path.join(fl_52mmdir, imname)

        if phase in ["val"]:
            # if imname.split(".")[0] in ["ip11pro_output_TextSR_renew_48",'ip11pro_output_TextSR_renew_15', 'ip11pro_output_textSR_211206_renew_59', 'ip11pro_output_xwm_testSR_211211_renew_4']:
            anno_path = os.path.join(gt_folder, "annos", imname.split(".")[0] + ".txt")
            anno_lines = open(anno_path, "r", encoding="gbk").readlines()
            if scale == 4:
                paths.append(
                    {"lq_path": impath_13mm, "gt_path": impath_52mm, "gt_annos": anno_lines, 'syn_degr': False}) # impath_52mm
            elif scale == 2:
                paths.append(
                    {"lq_path": impath_26mm, "gt_path": impath_52mm, "gt_annos": anno_lines, 'syn_degr': False})
                paths.append(
                    {"lq_path": impath_13mm, "gt_path": impath_26mm, "gt_annos": anno_lines, 'syn_degr': False})
            else:
                raise Exception("Sorry, Real-CE has no this scale", scale)


        else:
            anno_path = os.path.join(gt_folder, "trans_annos_52mm_renov", "res_" + imname.split(".")[0] + ".txt")
            anno_lines = open(anno_path, "r", encoding="utf-8").readlines()
            # print("anno_lines:", anno_lines)
            #if scale == 4:
            # paths.append(
            #         {"lq_path": impath_13mm, "gt_path": impath_52mm, 'syn_degr': False, "gt_annos": anno_lines})
            #paths.append(
            #         {"lq_path": impath_26mm, "gt_path": impath_52mm, 'syn_degr': False, "gt_annos": anno_lines})
            paths.append(
                    {"lq_path": impath_52mm, "gt_path": impath_52mm, 'syn_degr': True, "gt_annos": anno_lines})

    return paths


def get_ReCTS(opt, gt_folder):

    gtim_folder = os.path.join(gt_folder, "img")
    gtanno_folder = os.path.join(gt_folder, "gt")

    paths = []

    imlist = os.listdir(gtim_folder)

    lens = len(imlist)
    cnt = 0
    for imname in imlist:
        # print("ReCTS:", cnt, "/", lens)
        if imname.split(".")[-1] == "jpg":
            im_path = os.path.join(gtim_folder, imname)
            gt_anno_path = os.path.join(gtanno_folder, imname.split(".")[0] + ".json")

            gt_annos = json.load(open(gt_anno_path, "r"))
            gt_lines = gt_annos["lines"]
            # print("imname:", im_path)
            all_lines = []
            for line in gt_lines:
                line_str = ",".join([str(coord) for coord in line["points"]]) + "," + line["transcription"]
                all_lines.append(line_str)

            paths.append(
                {"lq_path": im_path, "gt_path": im_path, 'syn_degr': True, "gt_annos": all_lines})
        cnt += 1
    return paths


def get_LSVT(opt, gt_folder):

    gtim_folder_0 = os.path.join(gt_folder, "train_full_images_0")
    gtim_folder_1 = os.path.join(gt_folder, "train_full_images_1")
    gtanno_file = os.path.join(gt_folder, "train_full_labels.json")

    json_dict = json.load(open(gtanno_file, "r", encoding="utf-8"))

    paths = []

    imlist_0 = os.listdir(gtim_folder_0)
    imlist_1 = os.listdir(gtim_folder_1)

    imlist = [os.path.join(gtim_folder_0, imname) for imname in imlist_0]
    imlist.extend([os.path.join(gtim_folder_1, imname) for imname in imlist_1])

    for imname in imlist:
        im_path = imname
        gt_key = im_path.split("/")[-1].split(".")[0]
        gt_lines = json_dict[gt_key]

        all_lines = []
        for line in gt_lines:
            if not line["illegibility"]:
                trans = line["transcription"]
                points = np.array(line["points"]).astype(np.int).reshape(-1).tolist()
                line_str = ",".join([str(coord) for coord in points]) + "," + trans
                all_lines.append(line_str)
        paths.append(
            {"lq_path": im_path, "gt_path": im_path, 'syn_degr': True, "gt_annos": all_lines})

    return paths


def get_RealSR(opt, gt_folder, scale=4):

    cameras = ["Canon", "Nikon"]
    phase = "Train" if opt["phase"] == "train" else "Test"
    paths = []

    for cam in cameras:
        cam_dir = os.path.join(gt_folder, cam, phase, str(scale))
        imnames = os.listdir(cam_dir)

        imkeys = {}
        for imname in imnames:
            imkey = imname.split("_")[1]
            if not imkey in imkeys:
                imkeys[imkey] = 1
        # keylist = list(imkeys.keys())

        for imkey in imkeys:

            LR_path = os.path.join(cam_dir, cam + "_" + imkey + "_LR4.png")
            HR_path = os.path.join(cam_dir, cam + "_" + imkey + "_HR.png")

            paths.append({"lq_path": LR_path, "gt_path": HR_path, 'syn_degr': False, "gt_annos": []})
            paths.append({"lq_path": HR_path, "gt_path": HR_path, 'syn_degr': True, "gt_annos": []})
    return paths

@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwREC(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCEwREC, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []
        # print("self.gt_folders:", self.gt_folders)
        for gt_folder in self.gt_folders:

            print("gt_folder：", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
            elif "ReCTS" in gt_folder:
                self.paths.extend(get_ReCTS(self.opt, gt_folder))
            elif "LSVT" in gt_folder:
                self.paths.extend(get_LSVT(self.opt, gt_folder))
            elif "RealSR" in gt_folder:
                self.paths.extend(get_RealSR(self.opt, gt_folder))
            elif "TextZoom" in gt_folder:
                self.paths.extend(get_TextZoom(self.opt, gt_folder))
            elif "OtherDevice" in gt_folder:
                self.paths.extend(get_OtherDevice(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_lq

    def generate_mask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        if mode == "train":
            gt_shape = gt_image_np.shape[:2]
        else:
            gt_shape = gt_image_np.shape[1:]
        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            str_lines.append(items[-1])
            poly_lines.append(np.array(items[:8]).reshape(4, 2))

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            mask = (torch.tensor(mask).to(device))
        else:
            mask = (torch.tensor(mask))
        if mode == "train":
            mask = mask[..., None]
        else:
            mask = mask[None, ...]
        return (mask + 1e-10).float(), str_lines

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # if self.opt['phase'] == "train":
        #     img_lq, img_gt = apply_translation(img_lq, img_gt)

        if self.paths[index]['syn_degr']:
            img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        # print("img_lq:", img_lq.shape, img_gt.shape, np.unique(img_gt))

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        H_lq, W_lq = img_lq.shape[:2]
        img_gt = img_gt[:H_lq * scale, :W_lq * scale]
        # print("img_lq:", img_lq.shape)

        # augmentation for training
        if self.opt['phase'] == 'train':
            mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            gt_size = self.opt['gt_size']
            # random crop
            # print("before_cropped_mask:", img_gt.shape, mask.shape)
            img_gt, img_lq, mask = paired_random_cropwmask(img_gt, img_lq, mask, gt_size, scale, gt_path)
            # print("cropped_mask:", img_gt.shape, mask.shape)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}
        if self.opt['phase'] == 'train':
            # ret_data["gt_lines"] = self.paths[index]['gt_annos']
            ret_data["mask"] = mask.permute(2, 0, 1)

        if self.opt['phase'] in ["val"]:
            ret_data["gt_lines"] = self.paths[index]['gt_annos']
            mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            ret_data["mask"] = mask
            ret_data["gt_labels"] = gt_labels

        return ret_data

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwRECwLabelMap(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCEwRECwLabelMap, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []
        # print("self.gt_folders:", self.gt_folders)
        for gt_folder in self.gt_folders:

            print("gt_folder：", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
            elif "ReCTS" in gt_folder:
                self.paths.extend(get_ReCTS(self.opt, gt_folder))
            elif "LSVT" in gt_folder:
                self.paths.extend(get_LSVT(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC) * 255.0

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC) / 255.0

        return img_lq

    def generate_mask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        if mode == "train":
            gt_shape = gt_image_np.shape[:2]
        else:
            gt_shape = gt_image_np.shape[1:]
        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            str_lines.append(items[-1])
            poly_lines.append(np.array(items[:8]).reshape(4, 2))

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            mask = (torch.tensor(mask).to(device))
        else:
            mask = (torch.tensor(mask))
        if mode == "train":
            mask = mask[..., None]
        else:
            mask = mask[None, ...]
        return mask, str_lines


    def generate_labelmask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch.lower() if ch.lower() in LABEL_DICT else " "] for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), (90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            #if width > height:
            #    orient[1] = -orient[1]
            #else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 4
                    radius_w = ch_orient[0] / 4
                    l = int(max(0, ctr[0]-radius_w))
                    r = int(ctr[0]+radius_w)
                    t = int(max(0, ctr[1]-radius_h))
                    b = int(ctr[1]+radius_h)

                    label_idx_mask[t:b, l:r] = label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # if self.opt['phase'] == "train":
        #     img_lq, img_gt = apply_translation(img_lq, img_gt)

        if self.paths[index]['syn_degr']:
            img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        # print("img_lq:", img_lq.shape)

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        H_lq, W_lq = img_lq.shape[:2]
        img_gt = img_gt[:H_lq * scale, :W_lq * scale]
        # print("img_lq:", img_lq.shape)

        label_mask, _ = self.generate_labelmask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])

        '''
        if self.opt['phase'] == 'val':
            mask_dir = "label_mask"
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)
            lmask = label_mask[0].data.cpu().numpy()
            unique_ = np.unique(lmask)
            max = unique_[-1]
            normalized_masks = ((lmask / (float(max) + 1e-10)) * 255).astype(np.uint8)
            mask_name = os.path.join(mask_dir, "mask_" + gt_path.split("/")[-1])
            cv2.imwrite(mask_name, normalized_masks)
        '''
        # augmentation for training
        if self.opt['phase'] == 'train':
            mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            gt_size = self.opt['gt_size']
            # random crop
            # print("before_cropped_mask:", img_gt.shape, mask.shape)
            img_gt, img_lq, mask, label_mask = paired_random_cropwmasks(img_gt, img_lq, mask, label_mask, gt_size, scale, gt_path)

            # print("cropped_mask:", img_gt.shape, mask.shape)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        H_p, W_p = img_lq.shape[:2]
        label_mask = torch.nn.functional.interpolate(label_mask.unsqueeze(0), (H_p, W_p), mode="nearest").squeeze(0)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, "label_mask":label_mask}
        if self.opt['phase'] == 'train':
            # ret_data["gt_lines"] = self.paths[index]['gt_annos']
            ret_data["mask"] = mask.permute(2, 0, 1)

        if self.opt['phase'] in ["val"]:
            ret_data["gt_lines"] = self.paths[index]['gt_annos']
            mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            ret_data["mask"] = mask
            ret_data["gt_labels"] = gt_labels

        return ret_data

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwRECwLabelMapwCanny(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCEwRECwLabelMapwCanny, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []
        # print("self.gt_folders:", self.gt_folders)
        for gt_folder in self.gt_folders:

            print("gt_folder：", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
            elif "ReCTS" in gt_folder:
                self.paths.extend(get_ReCTS(self.opt, gt_folder))
            elif "LSVT" in gt_folder:
                self.paths.extend(get_LSVT(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def generate_canny_map(self, img):

        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_lq

    def generate_mask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        if mode == "train":
            gt_shape = gt_image_np.shape[:2]
        else:
            gt_shape = gt_image_np.shape[1:]
        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            str_lines.append(items[-1])
            poly_lines.append(np.array(items[:8]).reshape(4, 2))

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            mask = (torch.tensor(mask).to(device))
        else:
            mask = (torch.tensor(mask))
        if mode == "train":
            mask = mask[..., None]
        else:
            mask = mask[None, ...]
        return mask, str_lines

    def generate_labelmask_v2(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        char_imdir = "./char_array/"

        eng_alphabet = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        eng_dict = {eng_alphabet[idx]:idx for idx in range(len(eng_alphabet))}
        engnum_dir = "./engnum/"

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch.lower() if ch.lower() in LABEL_DICT else " "] for ch in line_str]
            char_list = [ch for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), (90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            #if width > height:
            #    orient[1] = -orient[1]
            #else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 4
                    radius_w = ch_orient[0] / 3
                    l = int(max(0, ctr[0]-radius_w))
                    r = int(ctr[0]+radius_w)
                    t = int(max(0, ctr[1]-radius_h))
                    b = int(ctr[1]+radius_h)

                    char_impath = os.path.join(char_imdir, str(label_list[idx]) + ".png")
                    if os.path.isfile(char_impath):
                        if r - l < 1 or b - t < 1:
                            # print("char_array:", r - l, b - t)
                            continue
                        char_array = cv2.imread(char_impath, 0)
                        char_array = cv2.resize(char_array, (r-l, b-t), interpolation=cv2.INTER_NEAREST)
                        char_array = (char_array > 0).astype(np.uint8)
                    elif char_list[idx] in eng_dict:
                        if r - l < 1 or b - t < 1:
                            print("char_array:", r - l, b - t)
                            continue
                        if eng_dict[char_list[idx]] < 1:
                            continue
                        char_impath = os.path.join(engnum_dir, str(eng_dict[char_list[idx]]) + ".png")
                        # print("char_impath:", char_impath)
                        char_array = cv2.imread(char_impath, 0)
                        char_array = cv2.resize(char_array, (r - l, b - t), interpolation=cv2.INTER_NEAREST)
                        char_array = (char_array > 0).astype(np.uint8)
                    else:
                        char_array = 1

                    label_idx_mask[t:b, l:r] = char_array * label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def generate_labelmask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch.lower() if ch.lower() in LABEL_DICT else " "] for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), (90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            # if width > height:
            #    orient[1] = -orient[1]
            # else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 4
                    radius_w = ch_orient[0] / 4
                    l = int(max(0, ctr[0] - radius_w))
                    r = int(ctr[0] + radius_w)
                    t = int(max(0, ctr[1] - radius_h))
                    b = int(ctr[1] + radius_h)

                    label_idx_mask[t:b, l:r] = label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # if self.opt['phase'] == "train":
        #     img_lq, img_gt = apply_translation(img_lq, img_gt)

        if self.paths[index]['syn_degr']:
            img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        # print("img_lq:", img_lq.shape)

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        H_lq, W_lq = img_lq.shape[:2]
        img_gt = img_gt[:H_lq * scale, :W_lq * scale]
        # print("img_lq:", img_lq.shape)

        label_mask, _ = self.generate_labelmask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
        '''
        if self.opt['phase'] == 'val':
            mask_dir = "label_mask"
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)
            lmask = label_mask[0].data.cpu().numpy()
            unique_ = np.unique(lmask)
            max = unique_[-1]
            normalized_masks = ((lmask / (float(max) + 1e-10)) * 255).astype(np.uint8)
            mask_name = os.path.join(mask_dir, "mask_" + gt_path.split("/")[-1])
            cv2.imwrite(mask_name, normalized_masks)
        '''
        # augmentation for training
        if self.opt['phase'] == 'train':
            mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            gt_size = self.opt['gt_size']
            # random crop
            # print("before_cropped_mask:", img_gt.shape, mask.shape)
            img_gt, img_lq, mask, label_mask = paired_random_cropwmasks(img_gt, img_lq, mask, label_mask, gt_size,
                                                                        scale, gt_path)

            # print("cropped_mask:", img_gt.shape, mask.shape)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        H_p, W_p = img_lq.shape[:2]
        label_mask = torch.nn.functional.interpolate(label_mask.unsqueeze(0), (H_p, W_p), mode="nearest").squeeze(0)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # Calculate the canny maps
        lr_canny_map = self.generate_canny_map(img_lq)
        hr_canny_map = self.generate_canny_map(img_gt)
        lr_canny_map = torch.tensor(lr_canny_map).unsqueeze(0)
        hr_canny_map = torch.tensor(hr_canny_map).unsqueeze(0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Concatenate the maps
        img_lq = torch.cat([img_lq, lr_canny_map], dim=0)
        img_gt = torch.cat([img_gt, hr_canny_map], dim=0)
        '''
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        '''
        ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, "label_mask": label_mask}
        if self.opt['phase'] == 'train':
            # ret_data["gt_lines"] = self.paths[index]['gt_annos']
            ret_data["mask"] = mask.permute(2, 0, 1)

        if self.opt['phase'] in ["val"]:
            ret_data["gt_lines"] = self.paths[index]['gt_annos']
            mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            ret_data["mask"] = mask
            ret_data["gt_labels"] = gt_labels

        return ret_data

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwRECwLabelMapwCannyv2(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCEwRECwLabelMapwCannyv2, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []
        # print("self.gt_folders:", self.gt_folders)
        for gt_folder in self.gt_folders:

            print("gt_folder：", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
            elif "ReCTS" in gt_folder:
                self.paths.extend(get_ReCTS(self.opt, gt_folder))
            elif "LSVT" in gt_folder:
                self.paths.extend(get_LSVT(self.opt, gt_folder))
            elif "OtherDevice" in gt_folder:
                self.paths.extend(get_OtherDevice(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def generate_canny_map(self, img):

        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC) * 255

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        # img_lq[img_lq > 255] = 255
        # img_lq[img_lq < 0] = 0

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC) / 255

        return img_lq

    def generate_mask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        #if mode == "train":
        gt_shape = gt_image_np.shape[:2]
        # else:
        #     gt_shape = gt_image_np.shape[1:]
        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            str_lines.append(items[-1])

            poly = np.array(items[:8]).reshape(4, 2)
            rbox = cv2.minAreaRect(poly.astype(np.float32))
            width = rbox[1][0]
            height = rbox[1][1]
            if abs(rbox[2]) > 45:
                width *= 0.75
            else:
                height *= 0.75
            rbox = (rbox[0], (width, height), rbox[2])
            shrinked_poly = cv2.boxPoints(rbox)
            poly_lines.append(shrinked_poly)

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            mask = (torch.tensor(mask).to(device))
        else:
            mask = (torch.tensor(mask))
        # if mode == "train":
        #     mask = mask[..., None]
        # else:
        mask = mask[None, ...]
        return mask, str_lines

    def generate_labelmask_v2(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        char_imdir = "./char_array/"

        eng_alphabet = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        eng_dict = {eng_alphabet[idx]:idx for idx in range(len(eng_alphabet))}
        engnum_dir = "./engnum/"

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch.lower() if ch.lower() in LABEL_DICT else " "] for ch in line_str]
            char_list = [ch for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), (90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            #if width > height:
            #    orient[1] = -orient[1]
            #else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 4
                    radius_w = ch_orient[0] / 3
                    l = int(max(0, ctr[0]-radius_w))
                    r = int(min(ctr[0]+radius_w, gt_shape[1]))
                    t = int(max(0, ctr[1]-radius_h))
                    b = int(min(ctr[1]+radius_h, gt_shape[0]))

                    char_impath = os.path.join(char_imdir, str(label_list[idx]) + ".png")
                    if os.path.isfile(char_impath):
                        if r - l < 1 or b - t < 1:
                            # print("char_array:", r - l, b - t)
                            continue
                        char_array = cv2.imread(char_impath, 0)
                        char_array = cv2.resize(char_array, (r-l, b-t), interpolation=cv2.INTER_NEAREST)
                        char_array = (char_array > 0).astype(np.uint8)
                    elif char_list[idx] in eng_dict:
                        if r - l < 1 or b - t < 1:
                            # print("char_array:", r - l, b - t)
                            continue
                        if eng_dict[char_list[idx]] < 1:
                            continue
                        char_impath = os.path.join(engnum_dir, str(eng_dict[char_list[idx]]) + ".png")
                        # print("char_impath:", char_impath)
                        char_array = cv2.imread(char_impath, 0)
                        char_array = cv2.resize(char_array, (r - l, b - t), interpolation=cv2.INTER_NEAREST)
                        char_array = (char_array > 0).astype(np.uint8)
                    else:
                        char_array = 1

                    label_idx_mask[t:b, l:r] = char_array * label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def generate_labelmask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch.lower() if ch.lower() in LABEL_DICT else " "] for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), (90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            # if width > height:
            #    orient[1] = -orient[1]
            # else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 4
                    radius_w = ch_orient[0] / 4
                    l = int(max(0, ctr[0] - radius_w))
                    r = int(ctr[0] + radius_w)
                    t = int(max(0, ctr[1] - radius_h))
                    b = int(ctr[1] + radius_h)

                    label_idx_mask[t:b, l:r] = label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # if self.opt['phase'] == "train":
        #     img_lq, img_gt = apply_translation(img_lq, img_gt)

        if self.paths[index]['syn_degr']:
            img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        label_mask, _ = self.generate_labelmask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
        mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
        if self.opt['phase'] == 'train':
            if self.opt.get('deformation', False):
                scale_factor = 0.5 + np.random.rand()
                # print("scale_before:", img_gt.shape, img_lq.shape, mask.shape, label_mask.shape)
                [img_gt, img_lq], [mask, label_mask] = deformation_v2([img_gt, img_lq], [mask.float(), label_mask.float()],
                                                                   scale_factor)
            if self.opt.get('synthetic_degr', False):
                try:
                    degradation_type = 0 #random.random()
                    if degradation_type > 0.6:  # real-esrgan
                        ##input should be BGR 0~1 numpy H*W*C
                        ##output is RGB 0~1 tensor
                        img_lq = real_esrgan_degradation(img_lq, insf=2).squeeze(
                            0).detach().numpy()  # output numpy c*h*w 0~1 RGB random.choice([2, 4]
                        img_lq = img_lq.transpose((1, 2, 0))  # transfer to h*w*c
                    elif degradation_type > 0.05:
                        ##input should be RGB 0~1 numpy H*W*C
                        ##output is RGB 0~1 numpy H*W*C
                        gt_tmp = img_lq# [:, :, ::-1]  # transfer to RGB
                        # print("BSRGAN Degr")
                        img_lq, _ = bsrgan_degradation(gt_tmp, sf=2)  # random.choice([4, 6]lq_patchsize=None) # RGB 0~1 numpy h*w*c
                        img_lq = img_lq.astype(np.float32)
                    else:
                        img_lq = img_lq# [:, :, ::-1]  # out RGB [:, :, ::-1]
                        pass
                except Exception as e:
                    # gt_tmp = text_img[:,:,::-1]#transfer to RGB
                    # img_lq, _ = bsrgan_degradation(img_lq, sf=2)#RGB 0~1 numpy h*w*c
                    # lq = lq.astype(np.float32)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(['error degradation', img_lq.shape, e, exc_type, fname, exc_tb.tb_lineno])
                    img_lq = np.ascontiguousarray(img_lq)  # BGR  # out RGB #[:, :, ::-1]

                # print("scale_after:", img_gt.shape, img_lq.shape, mask.shape, label_mask.shape, np.unique(img_gt), np.unique(img_lq))

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        H_lq, W_lq = img_lq.shape[:2]
        img_gt = img_gt[:H_lq * scale, :W_lq * scale]
        mask = mask[:, :H_lq * scale, :W_lq * scale]
        label_mask = label_mask[:, :H_lq * scale, :W_lq * scale]
        # print("img_lq:", img_lq.shape)

        '''
        if self.opt['phase'] == 'val':
            mask_dir = "label_mask"
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)
            lmask = label_mask[0].data.cpu().numpy()
            unique_ = np.unique(lmask)
            max = unique_[-1]
            normalized_masks = ((lmask / (float(max) + 1e-10)) * 255).astype(np.uint8)
            mask_name = os.path.join(mask_dir, "mask_" + gt_path.split("/")[-1])
            cv2.imwrite(mask_name, normalized_masks)
        '''

        # print("premask:", mask.shape)
        # augmentation for training
        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']
            # random crop
            # print("before_cropped_mask:", img_gt.shape, mask.shape)

            img_gt, img_lq, mask, label_mask = paired_random_cropwmasks(img_gt, img_lq, mask, label_mask, gt_size,
                                                                        scale, gt_path)

            # print("cropped_mask:", img_gt.shape, mask.shape)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        H_p, W_p = img_lq.shape[:2]
        label_mask = torch.nn.functional.interpolate(label_mask.unsqueeze(0), (H_p, W_p), mode="nearest").squeeze(0)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # Calculate the canny maps
        lr_canny_map = self.generate_canny_map(img_lq)
        hr_canny_map = self.generate_canny_map(img_gt)
        lr_canny_map = torch.tensor(lr_canny_map).unsqueeze(0)
        hr_canny_map = torch.tensor(hr_canny_map).unsqueeze(0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Concatenate the maps
        img_lq = torch.cat([img_lq, lr_canny_map], dim=0)
        img_gt = torch.cat([img_gt, hr_canny_map], dim=0)

        ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, "label_mask": label_mask}

        ret_data["mask"] = mask# .permute(2, 0, 1)
        if self.opt['phase'] == 'val':
            ret_data["gt_labels"] = gt_labels
            ret_data["gt_lines"] = self.paths[index]['gt_annos']

        # print("mask:", img_lq.shape, img_gt.shape, label_mask.shape, mask.shape)

        return ret_data

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwRECwLabelMapwCannyv3(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCEwRECwLabelMapwCannyv3, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []
        # print("self.gt_folders:", self.gt_folders)
        for gt_folder in self.gt_folders:

            print("gt_folder：", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
            elif "ReCTS" in gt_folder:
                self.paths.extend(get_ReCTS(self.opt, gt_folder))
            elif "LSVT" in gt_folder:
                self.paths.extend(get_LSVT(self.opt, gt_folder))
            elif "OtherDevice" in gt_folder:
                self.paths.extend(get_OtherDevice(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def generate_canny_map(self, img):

        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC) * 255

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        # img_lq[img_lq > 255] = 255
        # img_lq[img_lq < 0] = 0

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC) / 255

        return img_lq

    def generate_mask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        #if mode == "train":
        gt_shape = gt_image_np.shape[:2]
        # else:
        #     gt_shape = gt_image_np.shape[1:]
        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            str_lines.append(items[-1])

            poly = np.array(items[:8]).reshape(4, 2)
            rbox = cv2.minAreaRect(poly.astype(np.float32))
            width = rbox[1][0]
            height = rbox[1][1]
            # if abs(rbox[2]) > 45:
            #    width *= 0.75
            #else:
            #    height *= 0.75
            rbox = (rbox[0], (width, height), rbox[2])
            shrinked_poly = cv2.boxPoints(rbox)
            poly_lines.append(shrinked_poly)

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            mask = (torch.tensor(mask).to(device))
        else:
            mask = (torch.tensor(mask))
        # if mode == "train":
        #     mask = mask[..., None]
        # else:
        mask = mask[None, ...]
        return mask, str_lines

    def generate_labelmask_v2(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        char_imdir = "./char_array/"

        eng_alphabet = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
        eng_dict = {eng_alphabet[idx]:idx for idx in range(len(eng_alphabet))}
        engnum_dir = "./engnum/"

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch if ch in LABEL_DICT else " "] for ch in line_str]
            char_list = [ch for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), -(90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            #if width > height:
            #    orient[1] = -orient[1]
            #else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 2
                    radius_w = ch_orient[0] / 2
                    l = int(max(0, ctr[0]-radius_w))
                    r = int(min(ctr[0]+radius_w, gt_shape[1]))
                    t = int(max(0, ctr[1]-radius_h))
                    b = int(min(ctr[1]+radius_h, gt_shape[0]))

                    char_impath = os.path.join(char_imdir, str(label_list[idx]) + ".png")
                    if os.path.isfile(char_impath):
                        if r - l < 1 or b - t < 1:
                            # print("char_array:", r - l, b - t)
                            continue
                        char_array = cv2.imread(char_impath, 0)
                        char_array = cv2.resize(char_array, (r-l, b-t), interpolation=cv2.INTER_NEAREST)
                        char_array = (char_array > 0).astype(np.uint8)
                    elif char_list[idx] in eng_dict:
                        if r - l < 1 or b - t < 1:
                            # print("char_array:", r - l, b - t)
                            continue
                        if eng_dict[char_list[idx]] < 1:
                            continue
                        char_impath = os.path.join(engnum_dir, str(eng_dict[char_list[idx]]) + ".png")
                        # print("char_impath:", char_impath)
                        char_array = cv2.imread(char_impath, 0)
                        char_array = cv2.resize(char_array, (r - l, b - t), interpolation=cv2.INTER_NEAREST)
                        char_array = (char_array > 0).astype(np.uint8)
                    else:
                        char_array = 1

                    label_idx_mask[t:b, l:r] = char_array * label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def generate_labelmask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image
        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        label_idx_mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            line_str = items[-1]
            str_lines.append(line_str)
            poly_lines.append(np.array(items[:8]).reshape(4, 2))
            label_list = [LABEL_DICT[ch.lower() if ch.lower() in LABEL_DICT else " "] for ch in line_str]

            rect = cv2.minAreaRect(np.array(items[:8]).astype(np.int).reshape(4, 2))

            # get width and height of the detected rectangle
            if abs(rect[2]) > 45:
                width = rect[1][0]
                height = rect[1][1]
                rect = (rect[0], (height, width), (90 - abs(rect[2])))
            width = int(rect[1][0])
            height = int(rect[1][1])

            real_poly = cv2.boxPoints(rect)
            real_poly = np.int0(real_poly)
            lb, lt, rt, rb = real_poly[0], real_poly[1], real_poly[2], real_poly[3]
            # lt, rt, rb, lb,
            if width > height:
                pt_start = (lt + lb) / 2
                pt_end = (rt + rb) / 2
            else:
                pt_start = (lt + rt) / 2
                pt_end = (lb + rb) / 2

            orient = pt_end - pt_start

            # if width > height:
            #    orient[1] = -orient[1]
            # else:
            #    orient[0] = -orient[0]
            label_num = len(label_list)
            if label_num > 0:
                ch_orient = orient / label_num
                ch_ctrs = [pt_start + ch_orient * (cnt + 0.5) for cnt in range(label_num)]

                # Full the label mask
                for idx in range(len(ch_ctrs)):
                    ctr = ch_ctrs[idx]
                    radius_h = height / 4
                    radius_w = ch_orient[0] / 4
                    l = int(max(0, ctr[0] - radius_w))
                    r = int(ctr[0] + radius_w)
                    t = int(max(0, ctr[1] - radius_h))
                    b = int(ctr[1] + radius_h)

                    label_idx_mask[t:b, l:r] = label_list[idx]

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            label_idx_mask = (torch.tensor(label_idx_mask).to(device))
        else:
            label_idx_mask = (torch.tensor(label_idx_mask))

        return label_idx_mask.unsqueeze(0), str_lines

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # if self.opt['phase'] == "train":
        #     img_lq, img_gt = apply_translation(img_lq, img_gt)

        if self.paths[index]['syn_degr']:
            img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        label_mask_gt, _ = self.generate_labelmask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
        mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
        if self.opt['phase'] == 'train':
            if self.opt['deformation']:
                scale_factor = 0.5 + np.random.rand()
                # print("scale_before:", img_gt.shape, img_lq.shape, mask.shape, label_mask.shape)
                [img_gt, img_lq], [mask, label_mask_gt] = deformation_v2([img_gt, img_lq], [mask.float(), label_mask_gt.float()],
                                                                   scale_factor)
                # print("scale_after:", img_gt.shape, img_lq.shape, mask.shape, label_mask.shape)

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        H_lq, W_lq = img_lq.shape[:2]
        img_gt = img_gt[:H_lq * scale, :W_lq * scale]
        mask = mask[:, :H_lq * scale, :W_lq * scale]
        label_mask_gt = label_mask_gt[:, :H_lq * scale, :W_lq * scale]
        # print("img_lq:", img_lq.shape)

        '''
        if self.opt['phase'] == 'val':
            mask_dir = "label_mask"
            if not os.path.isdir(mask_dir):
                os.makedirs(mask_dir)
            lmask = label_mask[0].data.cpu().numpy()
            unique_ = np.unique(lmask)
            max = unique_[-1]
            normalized_masks = ((lmask / (float(max) + 1e-10)) * 255).astype(np.uint8)
            mask_name = os.path.join(mask_dir, "mask_" + gt_path.split("/")[-1])
            cv2.imwrite(mask_name, normalized_masks)
        '''

        # print("premask:", mask.shape)
        # augmentation for training
        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']
            # random crop
            # print("before_cropped_mask:", img_gt.shape, mask.shape)

            img_gt, img_lq, mask, label_mask_gt = paired_random_cropwmasks(img_gt, img_lq, mask, label_mask_gt, gt_size,
                                                                        scale, gt_path)

            # print("cropped_mask:", img_gt.shape, mask.shape)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        H_p, W_p = img_lq.shape[:2]
        label_mask = torch.nn.functional.interpolate(label_mask_gt.unsqueeze(0), (H_p, W_p), mode="nearest").squeeze(0)

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        # if self.opt['phase'] != 'train':
        #     img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # Calculate the canny maps
        lr_canny_map = self.generate_canny_map(img_lq)
        hr_canny_map = self.generate_canny_map(img_gt)
        lr_canny_map = torch.tensor(lr_canny_map).unsqueeze(0)
        hr_canny_map = torch.tensor(hr_canny_map).unsqueeze(0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Concatenate the maps
        img_lq = torch.cat([img_lq, lr_canny_map], dim=0)
        img_gt = torch.cat([img_gt, hr_canny_map], dim=0)

        ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path, "label_mask": label_mask, "label_mask_gt": label_mask_gt}

        ret_data["mask"] = mask# .permute(2, 0, 1)
        if self.opt['phase'] == 'val':
            ret_data["gt_labels"] = gt_labels
            ret_data["gt_lines"] = self.paths[index]['gt_annos']

        # print("mask:", img_lq.shape, img_gt.shape, label_mask.shape, mask.shape)

        return ret_data

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class PairedImageDatasetRealCEwRECCanny(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetRealCEwRECCanny, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folders, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.gt_folder = self.gt_folders

        if not type(self.gt_folder) == list:
            self.gt_folders = [self.gt_folders]

        self.paths = []
        # print("self.gt_folders:", self.gt_folders)
        for gt_folder in self.gt_folders:

            print("gt_folder：", gt_folder)

            if self.io_backend_opt['type'] == 'lmdb':
                self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
                self.io_backend_opt['client_keys'] = ['lq', 'gt']
                self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
            elif "RealCE" in gt_folder:
                self.paths.extend(get_RealCE(self.opt, gt_folder))
            elif "ReCTS" in gt_folder:
                self.paths.extend(get_ReCTS(self.opt, gt_folder))
            elif "LSVT" in gt_folder:
                self.paths.extend(get_LSVT(self.opt, gt_folder))

        self.nSamples = len(self.paths)
        print("nSamples:", self.nSamples)

    def __len__(self):
        return self.nSamples

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_lq

    def generate_mask(self, gt_image, gt_lines, mode="train"):

        input_type = 'Tensor' if torch.is_tensor(gt_image) else 'Numpy'

        if input_type == "Tensor":
            device = gt_image.device
            gt_image_np = gt_image.data.cpu().numpy()
            # gt_shape = gt_image_np.shape[1:]
        else:
            gt_image_np = gt_image

        gt_shape = gt_image_np.shape[:2]

        # print("gt_image_np:", gt_image_np.shape)
        mask = np.zeros(gt_shape).astype(np.uint8)
        str_lines = []
        poly_lines = []
        for i in range(len(gt_lines)):
            line = gt_lines[i]
            items = line.replace("\n", "").split(",")
            str_lines.append(items[-1])
            poly_lines.append(np.array(items[:8]).reshape(4, 2))

        poly_lines = np.array(poly_lines).astype(np.int)
        cv2.fillPoly(mask, poly_lines, 1)
        # print("poly_lines:", np.unique(mask), np.sum(mask))
        # print("mask_np:", mask.shape)
        if input_type == "Tensor":
            mask = (torch.tensor(mask).to(device))
        else:
            mask = (torch.tensor(mask))
        mask = mask[..., None]
        return mask, str_lines

    def generate_canny_map(self, img):

        img_255 = (img * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_255, cv2.COLOR_BGR2GRAY)
        # print("np.unique()", np.unique(img_gray))
        canny_map = cv2.Canny(img_gray, 100, 200)

        # cv2.imwrite("canny.jpg", canny_map)

        # to [0, 1]
        return (canny_map > 0).astype(np.float32)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        if self.paths[index]['syn_degr']:
            img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        # if self.opt['phase'] == 'train':
        #     img_lq, img_gt = apply_translation(img_lq, img_gt)

        # Downsample
        H, W = img_gt.shape[:2]
        H_lq, W_lq = img_lq.shape[:2]
        # if H == H_lq or W == W_lq:
        img_lq = cv2.resize(img_lq, (W // scale, H // scale), interpolation=cv2.INTER_CUBIC)
        H_lq, W_lq = img_lq.shape[:2]
        img_gt = img_gt[:H_lq * scale, :W_lq * scale]

        mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            cnt = 0
            while True:
                img_gt_crop, img_lq_crop, mask_crop = paired_random_cropwmask(img_gt, img_lq, mask, gt_size, scale, gt_path)
                mask_np = mask_crop.data.cpu().numpy()
                if cnt > 5:
                    img_gt = img_gt_crop
                    img_lq = img_lq_crop
                    mask = mask_crop
                    break
                # The sampled mask has a foreground label
                if np.sum(mask_np).shape[0] > 0.5:
                    img_gt = img_gt_crop
                    img_lq = img_lq_crop
                    mask = mask_crop
                    break
                cnt += 1
            # print("cropped_mask:", img_gt.shape, mask.shape)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # Calculate the canny maps
        lr_canny_map = self.generate_canny_map(img_lq)
        hr_canny_map = self.generate_canny_map(img_gt)
        lr_canny_map = torch.tensor(lr_canny_map).unsqueeze(0)
        hr_canny_map = torch.tensor(hr_canny_map).unsqueeze(0)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        # Concatenate the maps
        img_lq = torch.cat([img_lq, lr_canny_map], dim=0)
        img_gt = torch.cat([img_gt, hr_canny_map], dim=0)

        ret_data = {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path,
                    "mask": mask.permute(2, 0, 1)}

        if self.opt['phase'] in ["val"]:
            ret_data["gt_lines"] = self.paths[index]['gt_annos']
            # mask, gt_labels = self.generate_mask(img_gt, self.paths[index]['gt_annos'], self.opt['phase'])
            # ret_data["mask"] = mask
            ret_data["gt_labels"] = gt_labels

        return ret_data

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetSynthetic(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetSynthetic, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder = opt['dataroot_gt'] #, self.lq_folder, opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
        #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
        #                                                   self.opt['meta_info_file'], self.filename_tmpl)
        else:
            # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            # self.gt_folder = os.path.join(self.gt_folder, self.opt['phase'])

            fl_0 = os.path.join(self.gt_folder, "train_full_images_0")
            fl_1 = os.path.join(self.gt_folder, "train_full_images_1")

            self.paths = []

            imlist0 = os.listdir(fl_0)
            imlist0 = [os.path.join(fl_0, im) for im in imlist0]

            imlist1 = os.listdir(fl_1)
            imlist1 = [os.path.join(fl_1, im) for im in imlist1]

            imlist = imlist0 + imlist1

            # print("imlist:", len(imlist))

            self.paths = [{"gt_path": im} for im in imlist]

            '''
            for imname in imlist:
                impath_13mm = os.path.join(fl_13mmdir, imname)
                impath_26mm = os.path.join(fl_26mmdir, imname)
                impath_52mm = os.path.join(fl_52mmdir, imname)

                self.paths.append({"gt_path": impath_26mm})
                self.paths.append({"gt_path": impath_52mm})
                self.paths.append({"gt_path": impath_52mm})
            '''

    def degradation(self, image_hr, training):

        H, W, C = image_hr.shape

        # Down-sample
        img_lq = cv2.resize(image_hr, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        if random.random() > 0.3 or not training:
            img_lq = blur(img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = noisy("gauss", img_lq, not training)
        if random.random() > 0.3 or not training:
            img_lq = JPEG_compress(img_lq, not training)

        img_lq = cv2.resize(img_lq, (W, H), interpolation=cv2.INTER_CUBIC)

        return img_lq


    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        # lq_path = self.paths[index]['lq_path']
        # img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = img_gt.copy() # imfrombytes(img_bytes, float32=True)

        # Downsample to 1/2
        H, W = img_gt.shape[:2]
        img_lq = cv2.resize(img_lq, (W // 2, H // 2), interpolation=cv2.INTER_CUBIC)

        img_lq = self.degradation(img_lq, self.opt['phase'] == "train")

        H_l, W_l = img_lq.shape[:2]
        img_gt = cv2.resize(img_lq, (W_l * 2, H_l * 2), interpolation=cv2.INTER_CUBIC)

        # print("img_lq:", img_lq.shape)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': gt_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)

@DATASET_REGISTRY.register()
class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Downsample to 1/2
        H_g, W_g = img_gt.shape[:2]
        H_l, W_l = img_lq.shape[:2]
        if H_l == H_g or W_l == W_g:
            img_lq = cv2.resize(img_lq, (W_g // scale, H_g // scale), interpolation=cv2.INTER_CUBIC)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


@DATASET_REGISTRY.register()
class PairedImageDatasetEngSR(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageDatasetEngSR, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        imlist = os.listdir(self.gt_folder)
        self.imindices = [imname.split("_")[0] for imname in imlist]
        self.paths = []
        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
            # elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            #     self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #                                                   self.opt['meta_info_file'], self.filename_tmpl)
        else:

            for idx in self.imindices:
                gt_path = os.path.join(self.gt_folder, idx + "_orig.png")
                lq_path = os.path.join(self.lq_folder, idx + "_degr.png")
                self.paths.append({"lq_path":lq_path, "gt_path":gt_path})

        print("All samples:", len(self.paths))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)
        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Downsample to 1/2
        H_g, W_g = img_gt.shape[:2]
        H_l, W_l = img_lq.shape[:2]
        if H_l == H_g or W_l == W_g:
            img_lq = cv2.resize(img_lq, (W_g // scale, H_g // scale), interpolation=cv2.INTER_CUBIC)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)