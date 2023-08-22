import cv2
import numpy as np
import random
import os


def blur(image, testing=False):

    kernel_num = random.randint(1, 6) * 2 + 1
    if testing:
        kernel_num = 21
    image = cv2.GaussianBlur(image, (kernel_num, kernel_num), 1)

    return image

def noisy(noise_typ, image, testing=False):

    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 50
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        # print("gauss:", np.unique(gauss))
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row,col,ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out

    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


def JPEG_compress(image, testing=False):

    quality = random.randint(30, 80)
    if testing:
        quality = 50
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    ret_img = cv2.imdecode(encimg, 1)
    return ret_img