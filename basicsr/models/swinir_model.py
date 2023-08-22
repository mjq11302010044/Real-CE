import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
import math

@MODEL_REGISTRY.register()
class SwinIRModel(SRModel):

    def test(self):
        # print("SwinIR testing...")
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                # self.output = self.net_g_ema(img)
                self.output = self.tile_process_parallel(img, self.net_g_ema)
        else:
            self.net_g.eval()
            with torch.no_grad():
                # self.output = self.net_g(img)
                self.output = self.tile_process_parallel(img, self.net_g)
            self.net_g.train()

        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]

    def tile_process_parallel(self, img, model, tile_size=54, tile_pad=5):
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

        sub_patch = 8

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
            if len(patches) < 1:
                continue
            patches = torch.cat(patches, dim=0)
            decro_output_tile = model(patches)
            decro_output_patches.extend([decro_output_tile[i:i+1] for i in range(decro_output_tile.shape[0])])

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

        return output# / weighted_im

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

                output_tile = decro_output_tile[:, :, yl_shift * scale:yr_shift * scale, xl_shift * scale:xr_shift * scale]

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
