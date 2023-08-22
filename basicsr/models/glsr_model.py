import torch
import torch.nn.functional as F
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm
from copy import deepcopy

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
import math
import numpy as np
import cv2, os
import editdistance

@MODEL_REGISTRY.register()
class GLSRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(GLSRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self.mask = None
        self.label_mask = None
        # load pretrained models

        load_local_path = self.opt['path_local'].get('pretrain_network_g', None)
        if load_local_path is not None:
            param_key = self.opt['path_local'].get('param_key_g', 'params')
            self.load_local_network(self.net_g, load_local_path, self.opt['path_local'].get('strict_load_g', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # print("self.opt['path']['visualization']:", self.opt['path']['visualization'])

        if self.is_train:
            self.init_training_settings()

    def load_local_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """

        # print("load_path:", load_path)

        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path) # , map_location=lambda storage, loc: storage
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]

        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net["text_recovery_net." + k[7:]] = v
                load_net.pop(k)
            else:
                load_net["text_recovery_net." + k] = v
                load_net.pop(k)

        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    def torch_rotate_img(self, torch_image_batches, arc_batches, rand_offs, off_range=0.2):

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


        x_shift = torch.zeros_like(arc_batches)
        y_shift = torch.zeros_like(arc_batches)

        # print("device:", device)
        affine_matrix = torch.cat([a11.unsqueeze(1), a12.unsqueeze(1) * ratios_mul, x_shift.unsqueeze(1),
                                   a21.unsqueeze(1) / ratios_mul, a22.unsqueeze(1), y_shift.unsqueeze(1)], dim=1)
        affine_matrix = affine_matrix.reshape(N, 2, 3).to(device)

        affine_grid = F.affine_grid(affine_matrix, torch_image_batches.shape)
        distorted_batches = F.grid_sample(torch_image_batches, affine_grid)

        return distorted_batches

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        # define losses
        if train_opt.get('pixel_opt_local'):
            self.cri_pix_local = build_loss(train_opt['pixel_opt_local']).to(self.device)
        else:
            self.cri_pix_local = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('perceptual_opt_local'):
            self.cri_perceptual_local = build_loss(train_opt['perceptual_opt_local']).to(self.device)
        else:
            self.cri_perceptual_local = None

        if train_opt.get('gpt_opt'):
            self.cri_gpt = build_loss(train_opt['gpt_opt']).to(self.device)
        else:
            self.cri_gpt = None

        if train_opt.get('wtv_opt'):
            self.cri_wtv = build_loss(train_opt['wtv_opt']).to(self.device)
        else:
            self.cri_wtv = None

        if train_opt.get('seman_opt'):
            self.cri_seman = build_loss(train_opt['seman_opt']).to(self.device)
        else:
            self.cri_seman = None

        if train_opt.get('aux_loss'):
            self.loss_aux = True
        else:
            self.loss_aux = False

        if train_opt.get('tssim_opt'):
            self.loss_tssim = build_loss(train_opt['tssim_opt']).to(self.device)
        else:
            self.loss_tssim = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)
        if 'label_mask' in data:
            self.label_mask = data['label_mask'].to(self.device).float()
        if 'gt_lines' in data:
            self.gt_lines = data['gt_lines']

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        if self.loss_aux:
            self.output, self.emb_loss = self.net_g(self.lq)
        else:
            if not (self.mask is None):
                self.outputs = self.net_g(self.lq, self.gt, self.mask)
            else:
                self.outputs = self.net_g(self.lq)

        self.output, output_internal, global_canvas, cropped_srs, cropped_xs, cropped_ys, self.x_rec_priors, self.y_rec_priors = self.outputs

        # x_np = self.lq[:, :3].data.cpu().numpy()
        # print("sr model:", x_np.shape, np.unique(x_np))

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            if not (self.mask is None):
                l_pix = self.cri_pix(self.output, self.gt, self.mask)
            else:
                l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
            if not output_internal is None:
                if not (self.mask is None):
                    l_pix_internal = self.cri_pix(output_internal, self.gt, self.mask)
                else:
                    l_pix_internal = self.cri_pix(output_internal, self.gt)
                l_total += l_pix_internal
                loss_dict['l_pix_internal'] = l_pix_internal

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt, self.mask) * 0.5
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output[:, :], self.gt[:, :], self.mask)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

            if not output_internal is None:
                if not (self.mask is None):
                    l_percep_internal, l_style = self.cri_perceptual(output_internal, self.gt, self.mask)
                else:
                    l_percep_internal, l_style = self.cri_perceptual(output_internal, self.gt)
                l_total += l_percep_internal
                loss_dict['l_percep_internal'] = l_percep_internal

        if self.cri_pix_local:
            # if not (self.mask is None):
            #     l_pix_local = self.cri_pix_local(cropped_srs, cropped_ys, self.mask)
            # else:
            l_pix_local = self.cri_pix_local(cropped_srs, cropped_ys)
            l_total += l_pix_local
            loss_dict['l_pix_local'] = l_pix_local

        # perceptual loss
        if self.cri_perceptual_local:
            l_percep_local, l_style_local = self.cri_perceptual_local(cropped_srs, cropped_ys, self.mask)
            if l_percep_local is not None:
                l_total += l_percep_local
                loss_dict['l_percep_local'] = l_percep_local
            if l_style_local is not None:
                l_total += l_style_local
                loss_dict['l_style_local'] = l_style_local

        if self.cri_gpt:
            l_gpt = self.cri_gpt(self.output, self.gt, self.mask)
            l_total += l_gpt
            loss_dict['l_gpt'] = l_gpt

        if self.cri_seman:
            # l_seman = self.cri_seman(self.x_rec_priors, self.y_rec_priors)
            # l_total += l_seman
            # loss_dict['l_seman'] = l_seman
            pass

        if self.loss_aux:
            l_total += self.emb_loss.mean()#  * 0.1
            loss_dict['loss_emb'] = self.emb_loss.mean()# * 0.1

        if self.loss_tssim:

            if self.opt['train']["rotate_train"]:
                rotate_train = self.opt['train']["rotate_train"]
                # print("We are in rotate_train", self.args.rotate_train)
                batch_size = self.lq.shape[0]

                # create range
                angle_batch = np.random.rand(batch_size) * rotate_train * 2 - rotate_train
                arc = angle_batch / 180. * math.pi
                rand_offs = torch.tensor(np.random.rand(batch_size)).float()

                arc = torch.tensor(arc).float()

                # images_lr_origin = self.lq.clone()
                # images_hr_origin = self.gt.clone()

                # print("shape:", self.lq.shape)

                images_lr = self.torch_rotate_img(self.lq, arc, rand_offs)
                images_hr = self.torch_rotate_img(self.gt, arc, rand_offs)

                images_lr_ret = self.torch_rotate_img(images_lr.clone(), -arc, rand_offs)
                images_hr_ret = self.torch_rotate_img(images_hr.clone(), -arc, rand_offs)

                if self.loss_aux:
                    self.output_rot, self.emb_loss_rot = self.net_g(images_lr)
                else:
                    self.output_rot = self.net_g(images_lr)

                self.rot_output = self.torch_rotate_img(self.output, arc, rand_offs)

                loss_tssim = (1 - self.loss_tssim(self.output_rot, self.rot_output, images_hr).mean()) * 0.1
                loss_dict['loss_tssim'] = loss_tssim
                l_total += loss_tssim

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        H, W = self.lq.shape[2:]
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()

            with torch.no_grad():
                if not self.mask is None:
                    # self.outputs = self.net_g_ema(self.lq, None, self.mask)
                    self.outputs = self.tile_process_parallel(self.lq[:, :], self.gt[:, :], self.mask, self.net_g_ema)
                else:
                    self.outputs = self.net_g_ema(self.lq) #torch.nn.functional.interpolate(self.lq, (H*2, W*2), mode="bicubic")#

                self.output, self.output_internal, self.global_canvas, self.cropped_srs, self.cropped_x, _, self.x_rec_priors, self.y_rec_priors = self.outputs
        else:
            self.net_g.eval()
            with torch.no_grad():
                # self.output = self.net_g(self.lq)
                if not self.mask is None:
                    # self.outputs = self.net_g(self.lq, None, self.mask)
                    self.outputs = self.tile_process_parallel(self.lq[:, :], self.gt[:, :], self.mask, self.net_g)
                else:
                    self.outputs = self.net_g(self.lq) #torch.nn.functional.interpolate(self.lq, (H*2, W*2), mode="bicubic") #
                # self.output = self.tile_process(self.lq, self.net_g)
                # torch.empty_cache()

                self.output, self.output_internal, self.global_canvas, self.cropped_srs, self.cropped_x, _, self.x_rec_priors, self.y_rec_priors = self.outputs

            self.net_g.train()

    def tile_process(self, img, model, tile_size=54, tile_pad=5):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        # print("Go tile process...")
        scale = self.opt.get('scale', 1)

        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)

        # print("output_shape:", output_shape)

        # start with black image
        output = img.new_zeros(output_shape)

        weighted_im = img.new_zeros(output_shape)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                xl_shift = abs(input_start_x - tile_pad) if input_start_x - tile_pad < 0 else 0
                yl_shift = abs(input_start_y - tile_pad) if input_start_y - tile_pad < 0 else 0
                xr_shift = input_tile.shape[-1] + xl_shift if input_end_x + tile_pad > width else tile_size + tile_pad * 2
                yr_shift = input_tile.shape[-2] + yl_shift if input_end_y + tile_pad > height else tile_size + tile_pad * 2

                decro_input_tile = torch.zeros((batch, channel, tile_size + tile_pad * 2, tile_size + tile_pad * 2)).to(input_tile.device)
                decro_input_tile[:, :, yl_shift:yr_shift, xl_shift:xr_shift] = input_tile

                decro_output_tile = model(decro_input_tile)

                output_tile = decro_output_tile[:, :, yl_shift * 2:yr_shift * 2, xl_shift * 2:xr_shift * 2]

                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                # put tile into output image

                out_real = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                output_start_x_tile:output_end_x_tile]

                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] += out_real

                weighted_im[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] += torch.ones_like(out_real)

        return output / weighted_im

    def tile_process_parallel(self, img, img_hr, label_mask_, model, tile_size=54, tile_pad=5):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        # print("Go tile process...")
        scale = self.opt.get('scale', 1)

        batch, channel, height, width = img.shape
        batch, channel_mask, height_mask, width_mask = label_mask_.shape

        if height_mask != height or width_mask != width:
            label_mask = torch.nn.functional.interpolate(label_mask_, (height, width), mode='nearest')
            batch, channel_mask, height_mask, width_mask = label_mask.shape
        else:
            label_mask = label_mask_

        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)

        # print("label_mask:", height_mask, height)

        sub_patch = 4

        # print("output_shape:", output_shape)

        # start with black image
        output = img.new_zeros(output_shape)

        weighted_im = img.new_zeros(output_shape)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        input_patches = []
        input_labelmasks = []
        input_positions = []
        shift_positions = []
        gt_patches = []

        gt_patches = []

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                # input_tile_width = input_end_x - input_start_x
                # input_tile_height = input_end_y - input_start_y
                # tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                input_tile_mask = label_mask[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]
                gt_tile = img_hr[:, :, input_start_y_pad*scale:input_end_y_pad*scale, input_start_x_pad*scale:input_end_x_pad*scale]
                # if not gt_img is None:
                #     gt_tile = gt_img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                xl_shift = abs(input_start_x - tile_pad) if input_start_x - tile_pad < 0 else 0
                yl_shift = abs(input_start_y - tile_pad) if input_start_y - tile_pad < 0 else 0
                xr_shift = input_tile.shape[-1] + xl_shift if input_end_x + tile_pad > width else tile_size + tile_pad * 2
                yr_shift = input_tile.shape[-2] + yl_shift if input_end_y + tile_pad > height else tile_size + tile_pad * 2

                decro_input_tile = torch.zeros((batch, channel, tile_size + tile_pad * 2, tile_size + tile_pad * 2)).to(input_tile.device)
                decro_input_tile[:, :, yl_shift:yr_shift, xl_shift:xr_shift] = input_tile

                decro_inputmask_tile = torch.zeros((batch, channel_mask, tile_size + tile_pad * 2, tile_size + tile_pad * 2)).to(
                    input_tile.device)
                decro_inputmask_tile[:, :, yl_shift:yr_shift, xl_shift:xr_shift] = input_tile_mask

                decro_gt_tile = torch.zeros((batch, channel, (tile_size + tile_pad * 2) * scale, (tile_size + tile_pad * 2) * scale)).to(
                    input_tile.device)
                decro_gt_tile[:, :, yl_shift*scale:yr_shift*scale, xl_shift*scale:xr_shift*scale] = gt_tile

                input_patches.append(decro_input_tile)
                input_labelmasks.append(decro_inputmask_tile)
                gt_patches.append(decro_gt_tile)
                input_positions.append([input_start_x, input_end_x, input_start_y, input_end_y])
                shift_positions.append([xl_shift, yl_shift, xr_shift, yr_shift])

        sb_num = int(len(input_patches) // sub_patch) + 1

        decro_output_patches = []
        all_cropped_sr = []
        all_cropped_x = []
        all_global_canvas = []
        all_out_internal = []

        all_x_rec_priors = []
        all_y_rec_priors = []

        for patch_i in range(sb_num):

            patches = input_patches[patch_i * sub_patch:(patch_i+1) * sub_patch]
            masks = input_labelmasks[patch_i * sub_patch:(patch_i+1) * sub_patch]
            gt_p = gt_patches[patch_i * sub_patch:(patch_i+1) * sub_patch]
            if len(patches) < 1:
                continue
            patches = torch.cat(patches, dim=0)
            masks = torch.cat(masks, dim=0)
            gt_p = torch.cat(gt_p, dim=0)
            # print("process mask:", masks.shape)
            decro_output_tile, output_internal, global_canvas, cropped_srs, cropped_x, _, x_rec_priors, y_rec_priors = model(patches, gt_p, masks)
            decro_output_patches.extend([decro_output_tile[i:i+1] for i in range(decro_output_tile.shape[0])])
            all_cropped_sr.append(cropped_srs)
            all_cropped_x.append(cropped_x)
            all_global_canvas.append(global_canvas)
            all_out_internal.append(output_internal)
            all_x_rec_priors.append(x_rec_priors)
            all_y_rec_priors.append(y_rec_priors)

        for idx in range(len(decro_output_patches)):

            decro_output_tile = decro_output_patches[idx]
            xl_shift, yl_shift, xr_shift, yr_shift = shift_positions[idx]
            input_start_x, input_end_x, input_start_y, input_end_y = input_positions[idx]

            output_tile = decro_output_tile[:, :, yl_shift * scale:yr_shift * scale, xl_shift * scale:xr_shift * scale]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image

            out_real = output_tile[:, :, output_start_y_tile:output_end_y_tile,
            output_start_x_tile:output_end_x_tile]

            # print("out_real:", out_real.shape, output_start_y, output_end_y, output_start_x, output_end_x)

            output[:, :, output_start_y:output_end_y,
            output_start_x:output_end_x] = out_real

            # weighted_im[:, :, output_start_y:output_end_y,
            # output_start_x:output_end_x] += torch.ones_like(out_real)

        return output, all_out_internal, all_global_canvas, all_cropped_sr, all_cropped_x, None, all_x_rec_priors, all_y_rec_priors # / weighted_im


    def tile_process_parallel_(self, img, model, tile_size=54, tile_pad=5):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        # print("Go tile process...")
        scale = self.opt.get('scale', 1)

        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)

        sub_patch = 4

        # print("output_shape:", output_shape)

        # start with black image
        output = img.new_zeros(output_shape)

        weighted_im = img.new_zeros(output_shape)

        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        input_patches = []
        input_positions = []
        shift_positions = []

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                xl_shift = abs(input_start_x - tile_pad) if input_start_x - tile_pad < 0 else 0
                yl_shift = abs(input_start_y - tile_pad) if input_start_y - tile_pad < 0 else 0
                xr_shift = input_tile.shape[-1] + xl_shift if input_end_x + tile_pad > width else tile_size + tile_pad * 2
                yr_shift = input_tile.shape[-2] + yl_shift if input_end_y + tile_pad > height else tile_size + tile_pad * 2

                decro_input_tile = torch.zeros((batch, channel, tile_size + tile_pad * 2, tile_size + tile_pad * 2)).to(input_tile.device)
                decro_input_tile[:, :, yl_shift:yr_shift, xl_shift:xr_shift] = input_tile

                input_patches.append(decro_input_tile)
                input_positions.append([input_start_x, input_end_x, input_start_y, input_end_y])
                shift_positions.append([xl_shift, yl_shift, xr_shift, yr_shift])

        sb_num = int(len(input_patches) // sub_patch) + 1

        decro_output_patches = []
        for patch_i in range(sb_num):

            patches = input_patches[patch_i * sub_patch:(patch_i+1) * sub_patch]
            patches = torch.cat(patches, dim=0)
            decro_output_tile = model(patches)
            decro_output_patches.extend([decro_output_tile[i:i+1] for i in range(decro_output_tile.shape[0])])

        for idx in range(len(decro_output_patches)):

            decro_output_tile = decro_output_patches[idx]
            xl_shift, yl_shift, xr_shift, yr_shift = shift_positions[idx]
            input_start_x, input_end_x, input_start_y, input_end_y = input_positions[idx]

            output_tile = decro_output_tile[:, :, yl_shift * 2:yr_shift * 2, xl_shift * 2:xr_shift * 2]

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            # put tile into output image

            out_real = output_tile[:, :, output_start_y_tile:output_end_y_tile,
            output_start_x_tile:output_end_x_tile]

            output[:, :, output_start_y:output_end_y,
            output_start_x:output_end_x] += out_real

            weighted_im[:, :, output_start_y:output_end_y,
            output_start_x:output_end_x] += torch.ones_like(out_real)

        return output / weighted_im

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        save_recognition = self.opt['val'].get('save_recognition', False)
        comparison = self.opt['val'].get('comparison', None)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}
        if "recognition" in self.opt['val']['metrics'].keys():
            self.sr_rec_list = []
            self.gt_rec_list = []
        if "recognition_divide" in self.opt['val']['metrics'].keys():
            self.sr_rec = {
                "CHN":[],
                "ENG":[]
            }
            self.gt_rec = {
                "CHN": [],
                "ENG": []
            }

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            # print("Total:", )

            # cv2.imwrite("mask_" + str(idx) + ".png", self.mask.data.cpu().numpy().astype(np.uint8)[0][0] * 255)

            visuals = self.get_current_visuals()

            sr_img = tensor2img([visuals['result']])
            H, W = sr_img.shape[:2]
            img_bicubic = torch.nn.functional.interpolate(visuals['lq'], (H, W), mode="bicubic")
            lr_img = tensor2img([img_bicubic])
            #sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img[..., :3]
                del self.gt
            if 'mask' in visuals:
                # print("mask:", mask.shape)
                mask = tensor2img([visuals['mask']])
                metric_data['mask'] = mask
            if 'gt_lines' in visuals:
                # print("mask:", mask.shape)
                # mask = tensor2img([visuals['gt_lines']])
                metric_data['gt_lines'] = visuals['gt_lines']

            if comparison == "LR":
                sr_img = lr_img
            elif comparison == "HR":
                sr_img = gt_img
            metric_data['img'] = sr_img[..., :3] ###sr_img
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                    
                imwrite(sr_img, save_img_path)

                # self.cropped_srs, self.cropped_x

                N_batch = len(self.cropped_srs)

                if self.opt['is_train']:
                    cropped_dir = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}')
                else:
                    if self.opt['val']['suffix']:
                        cropped_dir = osp.join(self.opt['path']['visualization'], dataset_name,
                                               f'{img_name}_{self.opt["val"]["suffix"]}')
                    else:
                        cropped_dir = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}')
                if not os.path.isdir(cropped_dir):
                    os.makedirs(cropped_dir)

                for idx_batch in range(N_batch):
                    sr_patches = self.cropped_srs[idx_batch]

                    global_canvas_batch = self.global_canvas[idx_batch]
                    # print("global_canvas:", global_canvas_batch.shape, len(self.global_canvas), sr_patches.shape)

                    for i in range(global_canvas_batch.shape[0]):

                        global_canvas = global_canvas_batch[i]
                        max_value = torch.abs(global_canvas).max()
                        global_canvas = global_canvas / max_value
                        global_canvas_np = (global_canvas.permute(1, 2, 0) * 255).data.cpu().numpy()
                        # print("global_canvas_np:", np.unique(global_canvas_np))
                        global_canvas_np[global_canvas_np > 255] = 255
                        global_canvas_np[global_canvas_np < 0] = 0
                        cv2.imwrite(os.path.join(cropped_dir, str(idx_batch) + "_" + str(i) + "_global_canvas.png"),
                                    cv2.cvtColor(global_canvas_np.astype(np.uint8), cv2.COLOR_RGB2BGR))

                    for i in range(sr_patches.shape[0]):

                        sr_im = sr_patches[i]
                        lr_im = self.cropped_x[idx_batch][i]

                        sr_imnp = (sr_im.permute(1, 2, 0) * 255).data.cpu().numpy()
                        lr_imnp = (lr_im.permute(1, 2, 0) * 255).data.cpu().numpy()

                        # print("sr_imnp:", np.unique(sr_imnp))

                        sr_imnp[sr_imnp > 255] = 255
                        sr_imnp[sr_imnp < 0] = 0
                        lr_imnp[lr_imnp > 255] = 255
                        lr_imnp[lr_imnp < 0] = 0

                        cv2.imwrite(os.path.join(cropped_dir, str(idx_batch) + "_" + str(i) + "sr.png"), cv2.cvtColor(sr_imnp.astype(np.uint8), cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(cropped_dir, str(idx_batch) + "_" + str(i) + "lr.png"), cv2.cvtColor(lr_imnp.astype(np.uint8), cv2.COLOR_RGB2BGR))

            if save_recognition:
                if self.opt['is_train']:
                    save_rec_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.txt')
                else:
                    if self.opt['val']['suffix']:
                        save_rec_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.txt')
                    else:
                        save_rec_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.txt')

            if with_metrics:
                # calculate metrics
                rec_cnt = 0
                # print("metrics:", self.opt['val']['metrics'].keys())
                for name, opt_ in self.opt['val']['metrics'].items():
                    if "recognition" in name:
                        # recognition performs only once
                        if rec_cnt > 0:
                            continue
                        sr_rec, gt_str = calculate_metric(metric_data, opt_)
                        rec_cnt += 1
                        if "recognition" in self.opt['val']['metrics'].keys():
                            if type(sr_rec) == dict:
                                for key in sr_rec:
                                    self.sr_rec_list.extend(sr_rec[key])
                                    self.gt_rec_list.extend(gt_str[key])

                                    if save_recognition:
                                        rec_f = open(save_rec_path, "a+")
                                        for i in range(len(sr_rec[key])):
                                            sr_, gt_ = sr_rec[key][i], gt_str[key][i]
                                            rec_f.write(sr_ + "\t" + gt_ + "\n")
                                        rec_f.close()
                            else:
                                self.sr_rec_list.extend(sr_rec)
                                self.gt_rec_list.extend(gt_str)
                        if "recognition_divide" in self.opt['val']['metrics'].keys():
                            # sr_rec, gt_str = calculate_metric(metric_data, opt_)
                            for key in sr_rec:
                                self.sr_rec[key].extend(sr_rec[key])
                                self.gt_rec[key].extend(gt_str[key])
                    elif "ned" in name:
                        pass
                    else:
                        self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                if metric == "recognition":
                    sr_list = self.sr_rec_list
                    gt_list = self.gt_rec_list
                    cnt = 0
                    for i in range(len(sr_list)):
                        if sr_list[i] == gt_list[i]:
                            cnt += 1
                        if "ned" in self.metric_results.keys():
                            pred, gt = sr_list[i], gt_list[i]
                            max_len = max(len(pred), len(gt))
                            self.metric_results["ned"] += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
                    self.metric_results[metric] = cnt / float(len(sr_list) + 1e-10)
                    if "ned" in self.metric_results.keys():
                        self.metric_results["ned"] /= float(len(sr_list) + 1e-10)
                    # self.metric_results

                    self.sr_rec_list = []
                    self.gt_rec_list = []

                elif metric == "recognition_divide":
                    sr_list = self.sr_rec
                    gt_list = self.gt_rec
                    cnt = {
                        "CHN": 0,
                        "ENG": 0
                    }
                    for key in sr_list:
                        for i in range(len(sr_list[key])):
                            if sr_list[key][i] == gt_list[key][i]:
                                cnt[key] += 1
                            if ("ned_" + key) in self.metric_results.keys():
                                pred, gt = sr_list[key][i], gt_list[key][i]
                                max_len = max(len(pred), len(gt))
                                self.metric_results["ned_" + key] += (max_len - editdistance.eval(pred, gt)) / float(max_len + 1e-10)
                        self.metric_results["recognition_" + key] = cnt[key] / float(len(sr_list[key]) + 1e-10)
                        if "ned_" + key in self.metric_results.keys():
                            self.metric_results["ned_" + key] /= float(len(sr_list[key]) + 1e-10)
                        # self.metric_results
                        # print("cnt", cnt, len(sr_list[key]))
                    self.sr_rec = {
                        "CHN": [],
                        "ENG": []
                    }
                    self.gt_rec = {
                        "CHN": [],
                        "ENG": []
                    }

                elif "ned" in metric:
                    pass
                elif "recognition" in metric:
                    pass
                else:
                    self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'

        log_table_path = "log_table_.csv"
        self.log_table_f = open(log_table_path, "a+")

        value_str = str(self.best_metric_results[dataset_name]["recognition"]["iter"]) + ","
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            value_str += f'{value:.4f}' + ","
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'
        value_str = value_str[:-1] + "\n"

        print("value_str:", value_str)

        self.log_table_f.write(value_str)
        self.log_table_f.close()

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        if hasattr(self, 'mask') and not (self.mask is None):
            out_dict['mask'] = self.mask.detach().cpu()
        if hasattr(self, 'gt_lines'):
            out_dict['gt_lines'] = self.gt_lines
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
