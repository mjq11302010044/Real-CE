import cv2
import numpy as np
import random
from scipy import ndimage
import imgaug.augmenters as ia
from scipy.interpolate import interp2d
from .unprocess import unprocess, random_noise_levels, add_noise
from .process import process
import tensorflow as tf
import glob
import os


def get_degrade_seq():
    degrade_seq = []
    need_shift = False
    global_sf = None

    # -----------------------
    # isotropic gaussian blur
    # -----------------------
    B_iso = {
        "mode": "blur",
        "kernel_size": random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
        "is_aniso": False,
        "sigma": random.uniform(0.1, 2.8),
    }
    degrade_seq.append(B_iso)

    # -------------------------
    # anisotropic gaussian blur
    # -------------------------
    B_aniso = {
        "mode": "blur",
        "kernel_size": random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
        "is_aniso": True,
        "x_sigma": random.uniform(0.5, 8),
        "y_sigma": random.uniform(0.5, 8),
        "rotation": random.uniform(0, 180)
    }
    degrade_seq.append(B_aniso)

    # -----------
    # down sample
    # -----------
    B_down = {
        "mode": "down",
        "sf": random.uniform(2, 4)
    }
    mode = random.randint(1, 4)
    if mode == 1:
        B_down["down_mode"] = "nearest"
        B_down["sf"] = random.choice([2, 4])
        need_shift = True
    elif mode == 2:
        B_down["down_mode"] = "bilinear"
    elif mode == 3:
        B_down["down_mode"] = "bicubic"
    elif mode == 4:
        down_mode = random.choice(["bilinear", "bicubic"])
        up_mode = random.choice(["bilinear", "bicubic"])
        up_sf = random.uniform(0.5, B_down["sf"])
        B_down["down_mode"] = down_mode
        B_down["sf"] = B_down["sf"] / up_sf
        B_up = {
            "mode": "down",
            "sf": up_sf,
            "down_mode": up_mode
        }
        degrade_seq.append(B_up)
    degrade_seq.append(B_down)
    global_sf = B_down["sf"]

    # --------------
    # gaussian noise
    # --------------
    B_noise = {
        "mode": "noise",
        "noise_level": random.randint(1, 25)
    }
    degrade_seq.append(B_noise)

    # ----------
    # jpeg noise
    # ----------
    if random.randint(1, 4) <= 3:
        B_jpeg = {
            "mode": "jpeg",
            "qf": random.randint(30, 95)
        }
        degrade_seq.append(B_jpeg)

    # -------------------
    # Processed camera sensor noise
    # -------------------
    if random.randint(1, 4) <= 4:
        B_camera = {
            "mode": "camera",
        }
        degrade_seq.append(B_camera)

    # -------
    # shuffle
    # -------
    random.shuffle(degrade_seq)

    # ---------------
    # last jpeg noise
    # ---------------
    B_jpeg_last = {
        "mode": "jpeg",
        "qf": random.randint(30, 95)
    }
    degrade_seq.append(B_jpeg_last)

    # --------------------
    # restore correct size
    # --------------------
    B_restore = {
        "mode": "restore",
        "sf": global_sf,
        "need_shift": need_shift
    }
    degrade_seq.append(B_restore)
    return degrade_seq


def degradation_pipeline(img):
    h, w, c = img.shape
    degrade_seq = get_degrade_seq()
    # print_degrade_seg(degrade_seq)
    for degrade_dict in degrade_seq:
        mode = degrade_dict["mode"]
        if mode == "blur":
            img = get_blur(img, degrade_dict)
        elif mode == "down":
            img = get_down(img, degrade_dict)
        elif mode == "noise":
            img = get_noise(img, degrade_dict)
        elif mode == 'jpeg':
            img = get_jpeg(img, degrade_dict)
        elif mode == 'camera':
            img = get_camera(img, degrade_dict)
        elif mode == 'restore':
            img = get_restore(img, h, w, degrade_dict)
    return img


def get_blur(img, degrade_dict):
    k_size = degrade_dict["kernel_size"]
    if degrade_dict["is_aniso"]:
        sigma_x = degrade_dict["x_sigma"]
        sigma_y = degrade_dict["y_sigma"]
        angle = degrade_dict["rotation"]
    else:
        sigma_x = degrade_dict["sigma"]
        sigma_y = degrade_dict["sigma"]
        angle = 0

    kernel = np.zeros((k_size, k_size))
    d = k_size // 2
    for x in range(-d, d+1):
        for y in range(-d, d+1):
            kernel[x+d][y+d] = get_kernel_pixel(x, y, sigma_x, sigma_y)
    M = cv2.getRotationMatrix2D((k_size//2, k_size//2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (k_size, k_size))
    kernel = kernel / np.sum(kernel)

    # kernel = kernel*255/np.max(kernel)
    # kernel = kernel.astype(np.uint8).reshape((k_size, k_size, 1))
    # cv2.imwrite("test.png", kernel)
    img = ndimage.filters.convolve(img, np.expand_dims(kernel, axis=2), mode='reflect')

    return img


def get_down(img, degrade_dict):
    sf = degrade_dict["sf"]
    mode = degrade_dict["down_mode"]
    h, w, c = img.shape
    if mode == "nearest":
        img = img[0::sf, 0::sf, :]
    elif mode == "bilinear":
        new_h, new_w = int(h/sf)//2*2, int(w/sf)//2*2
        img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_LINEAR)
    elif mode == "bicubic":
        new_h, new_w = int(h/sf)//2*2, int(w/sf)//2*2
        img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
    return img


def get_noise(img, degrade_dict):
    noise_level = degrade_dict["noise_level"]
    img = img + np.random.normal(0, noise_level, img.shape)
    return img


def get_jpeg(img, degrade_dict):
    qf = degrade_dict["qf"]
    trans = ia.JpegCompression(compression=qf)
    degrade_function = lambda x: trans.augment_image(x)
    img = degrade_function(img.astype(np.uint8))
    return img


def get_camera(img, degrade_dict):
    img = tf.convert_to_tensor(img, dtype=tf.float32) / 255.
    deg_img, features = unprocess(img)

    shot_noise, read_noise = random_noise_levels()
    deg_img = add_noise(deg_img, shot_noise, read_noise)

    deg_img = tf.expand_dims(deg_img, 0)
    features['red_gain'] = tf.expand_dims(features['red_gain'], axis=0)
    features['blue_gain'] = tf.expand_dims(features['blue_gain'], axis=0)
    features['cam2rgb'] = tf.expand_dims(features['cam2rgb'], axis=0)
    # print(features['red_gain'], features['blue_gain'], features['cam2rgb'])
    deg_img = process(deg_img, features['red_gain'], features['blue_gain'], features['cam2rgb'])
    # print(deg_img.shape)
    deg_img = tf.squeeze(deg_img)
    deg_img = tf.saturate_cast(deg_img * 255 + 0.5, tf.uint8)
    # print(deg_img.shape, type(deg_img))
    deg_img = np.array(deg_img).astype(np.uint8)
    # img = cv2.cvtColor(deg_img, cv2.COLOR_BGR2RGB)
    return deg_img


def get_restore(img, h, w, degrade_dict):
    need_shift = degrade_dict["need_shift"]
    sf = degrade_dict["sf"]
    img = cv2.resize(img, (h, w), interpolation=cv2.INTER_CUBIC)
    if need_shift:
        img = shift_pixel(img, int(sf))
    return img


def get_kernel_pixel(x, y, sigma_x, sigma_y):
    return 1/(2*np.pi*sigma_x*sigma_y)*np.exp(-((x*x/(2*sigma_x*sigma_x))+(y*y/(2*sigma_y*sigma_y))))


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: WxHxC or WxH
        sf: scale factor
        upper_left: shift direction
    """
    h, w = x.shape[:2]
    shift = (sf-1)*0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w-1)
    y1 = np.clip(y1, 0, h-1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


def print_degrade_seg(degrade_seq):
    for degrade_dict in degrade_seq:
        print(degrade_dict)


if __name__ == "__main__":
    test_blur = {
        "mode": "blur",
        "kernel_size": 21,  # random.choice([7, 9, 11, 13, 15, 17, 19, 21]),
        "is_aniso": True,
        "x_sigma": 0.1,  # random.uniform(0.5, 8),
        "y_sigma": 0.1,  # random.uniform(0.5, 8),
        "rotation": random.uniform(0, 5)
    }
    test_down = {
        "mode": "down",
        "sf": 2,
        "down_mode": "nearest"
    }
    test_noise = {
        "mode": "noise",
        "noise_level": 23
    }
    test_jpeg = {
        "mode": "jpeg",
        "qf": 30
    }
    test_camera = {
        "mode": "camera",
    }
    test_restore = {
        "mode": "restore",
        "sf": 2,
        "need_shift": False
    }
    img = cv2.imread("./tiny_test/65024.png")
    # h, w, c = img.shape
    # blur_img = get_blur(img, test_blur)
    # down_img = get_down(img, test_down)
    # noise_img = get_noise(img, test_noise)
    # jpeg_img = get_jpeg(img, test_jpeg)
    # camera_img = get_camera(img, test_camera)
    # restore_img = get_restore(down_img, h, w, test_restore)
    restore_img = degradation_pipeline(img)
    cv2.imwrite("./deg.png", restore_img)
    # cv2.imwrite("./deg.png", camera_img)

    # img_list = glob.glob('./tiny_test/*.png')
    # save_dir = "./tiny_test" + '_prac'
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    # for img_path in img_list:
    #     img = cv2.imread(img_path)
    #     img_name = os.path.basename(img_path)
    #     # print(img_name)
    #     img = degradation_pipeline(img)
    #     cv2.imwrite(os.path.join(save_dir, img_name), img)