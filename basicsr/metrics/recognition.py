import numpy as np
import os
from easydict import EasyDict
import basicsr.metrics.crnn as crnn
import torch
from torch.autograd import Variable
from basicsr.utils.registry import METRIC_REGISTRY
import cv2
import string
import math
# import levensteindistance

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

# if args.CHNSR:
opt['character'] = open("/home/majianqi/workspace/BasicSR/basicsr/metrics/al_chinese.txt", 'r').readlines()[0].replace("\n", "")
opt["num_class"] = len(opt['character'])
opt = EasyDict(opt)

def str_filt(str_, voc_type="benchmark"):
    alpha_dict = {
        'digit': string.digits,
        'lower': string.digits + string.ascii_lowercase,
        'upper': string.digits + string.ascii_letters,
        'all':   string.digits + string.ascii_letters + string.punctuation,
        'chinese': open("/home/majianqi/workspace/BasicSR/basicsr/metrics/al_chinese.txt", "r").readlines()[0].replace("\n", ""),
        'benchmark': open("/home/majianqi/workspace/BasicSR/basicsr/metrics/benchmark.txt", "r").readlines()[0].replace("\n", "")
    }
    if voc_type == 'lower':
        str_ = str_.lower()

    if voc_type in ['chinese', 'benchmark']: # Chinese character only
        new_str = ""
        for ch in str_:
            # if '\u4e00' <= ch <= '\u9fa5' or ch in string.digits + string.ascii_letters:
            new_str += ch
        str_ = new_str
    if type(str_) == list:
        str_ = str_[0]
    for char in str_:
        if char not in alpha_dict[voc_type]: #voc_type
            str_ = str_.replace(char, '')
    return str_

def get_string_crnn(outputs_, use_chinese, alphabet='-0123456789abcdefghijklmnopqrstuvwxyz'):
    outputs = outputs_.permute(1, 0, 2).contiguous()
    predict_result = []

    if use_chinese:
        # alphabet = open("basicsr/metrics/al_chinese.txt", 'r').readlines()[0].replace("\n", "")
        alphabet = list(open("/home/majianqi/workspace/BasicSR/basicsr/metrics/benchmark.txt", "r").readlines()[0].replace("\n", ""))
        alphabet.insert(0, ' ')

    for output in outputs:
        max_index = torch.max(output, 1)[1]

        out_str = ""
        last = ""
        for i in max_index:
            if alphabet[i] != last:
                if i != 0:
                    out_str += alphabet[i]
                    last = alphabet[i]
                else:
                    last = ""

        predict_result.append(out_str)
    return predict_result

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

def CRNN_ENG_init(recognizer_path=None, opt=None):

    model = crnn.CRNN_ENG(32, 1, 37, 256)
    model = model.to(global_device)
    # cfg = self.config.TRAIN
    # aster_info = AsterInfo(cfg.voc_type)
    model_path = recognizer_path if not recognizer_path is None else opt.saved_model_eng
    print('loading pretrained TPG model from %s' % model_path)
    stat_dict = torch.load(model_path)

    model_keys = model.state_dict().keys()
    # print("state_dict:", len(stat_dict))
    if type(stat_dict) == list:
        print("state_dict:", len(stat_dict))
        stat_dict = stat_dict[0]  # .state_dict()
    # load_keys = stat_dict.keys()

    if recognizer_path is None:
        # model.load_state_dict(stat_dict)
        load_keys = stat_dict.keys()
        man_load_dict = model.state_dict()
        for key in stat_dict:
            if not key.replace("module.", "") in man_load_dict:
                print("Key not match", key, key.replace("module.", ""))
            man_load_dict[key.replace("module.", "")] = stat_dict[key]
        model.load_state_dict(man_load_dict)
    else:
        # model = stat_dict
        model.load_state_dict(stat_dict)
    return model

print("Initializing recognizer...")
recognizer = CRNN_init(opt=opt)
recognizer_ENG = CRNN_ENG_init(opt=opt)
print("done")

recognizers = {
    "CHN":recognizer,
    "ENG":recognizer_ENG
}

def parse_ENG_data(imgs_input_, ratio_keep=True):

    in_width = 512

    if ratio_keep:
        real_height, real_width = imgs_input_.shape[2:]
        ratio = real_width / float(real_height)

        # if ratio > 3:
        in_width = max(min(int(ratio * 32), 1024), 16)
    imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic') / float(255)

    # print("imgs_input:", imgs_input.shape)

    R = imgs_input[:, 0:1, :, :]
    G = imgs_input[:, 1:2, :, :]
    B = imgs_input[:, 2:3, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B
    return tensor


def parse_CRNN_data(imgs_input_, ratio_keep=True):

    in_width = 512

    if ratio_keep:
        real_height, real_width = imgs_input_.shape[2:]
        ratio = real_width / float(real_height)

        # if ratio > 3:
        in_width = max(min(int(ratio * 32), 1024), 16)
    imgs_input = torch.nn.functional.interpolate(imgs_input_, (32, in_width), mode='bicubic')

    return imgs_input

data_parser = {
    "CHN":parse_CRNN_data,
    "ENG":parse_ENG_data
}


def recognition_loss(img_sr, img_hr, lang="CHN", eval=False):
    # print("lang:", lang)
    # im_sr = img_sr
    # im_sr_th = torch.tensor(im_sr).permute(2, 0, 1).unsqueeze(0).float()
    processed_sr = data_parser[lang](img_sr)
    processed_hr = data_parser[lang](img_hr)
    # with torch.no_grad():
    # if eval and lang=="ENG":
    # recognizers[lang].eval()
    sr_output = recognizers[lang](processed_sr.to(global_device))
    hr_output = recognizers[lang](processed_hr.to(global_device))
    print("sr_output:", sr_output.shape)
    # if eval and lang=="ENG":
    # recognizers[lang].train()
    # sr_prediction = get_string_crnn(sr_output, lang==
    sr_log = sr_output.log_softmax(dim=1)
    hr_log = hr_output.log_softmax(dim=1)
    hr_prob = hr_output.softmax(dim=1)
    return hr_prob * (hr_log - sr_log)


def recognition_processing(img, lang="CHN", eval=False):
    # print("lang:", lang)
    im_sr = img
    im_sr_th = torch.tensor(im_sr).permute(2, 0, 1).unsqueeze(0).float()
    processed_sr = data_parser[lang](im_sr_th)
    with torch.no_grad():
        if eval and lang=="ENG":
            recognizers[lang].eval()
        sr_output = recognizers[lang](processed_sr.to(global_device))
        if eval and lang=="ENG":
            recognizers[lang].train()
    sr_prediction = get_string_crnn(sr_output, lang=="CHN")
    return sr_prediction


def get_polys(annos):
    # print (annos[0])
    polys = np.array([np.array(anno[0].split(",")[:8]).astype(np.int) for anno in annos]).reshape(-1, 4, 2)
    gt_strs = [(",".join((anno[0].split(",")[8:]))).replace("\n", "") for anno in annos]
    return polys, gt_strs

debug_visdir = "cropped_ims"
if not os.path.isdir(debug_visdir):
    os.makedirs(debug_visdir)

def crop_images(global_image, polys):
    # img = cv2.imread("big_vertical_text.jpg")
    # points for test.jpg
    all_warpped = []
    # print("polys:", polys)
    for idx in range(len(polys)):
        img = global_image.copy()
        cnt = polys[idx]
        # print("shape of cnt: {}".format(cnt.shape))
        # print("cnt:", cnt)
        rect = cv2.minAreaRect(cnt)
        #print("rect: {}".format(rect))

        # the order of the box points: bottom left, top left, top right,
        # bottom right

        # get width and height of the detected rectangle
        if abs(rect[2]) > 45:
            width = rect[1][0]
            height = rect[1][1]
            rect = (rect[0], (height, width), -(90 - abs(rect[2]))) ############
        # rect = (rect[0], (rect[1][0], rect[1][1] * 1.1), rect[2])
        width = int(rect[1][0])
        height = int(rect[1][1])

        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # print("bounding box: {}".format(box))

        src_pts = box.astype("float32")
        # coordinate of the points in box points after the rectangle has been
        # straightened
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(img, M, (width, height))
        all_warpped.append(warped)
        # cv2.waitKey(0)

    return all_warpped


def get_lang(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return "CHN"
    return "ENG"


@METRIC_REGISTRY.register()
def calculate_recognition(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = []
    gt_strs_filt = []
    cnt = 0
    all_langs = []
    for im in sr_warpped:
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        rec = recognition_processing(im, lang)
        sr_predictions.append(str_filt(rec, voc_types[lang]))
        gt_str = str_filt(gt_strs[cnt], voc_types[all_langs[cnt]]) # all to be lower
        gt_strs_filt.append(gt_str)
        print(str_filt(rec, voc_types[lang]), "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
        cnt += 1
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt


@METRIC_REGISTRY.register()
def calculate_recognition_case_insensitive(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = []
    gt_strs_filt = []
    cnt = 0
    all_langs = []
    for im in sr_warpped:
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        rec = recognition_processing(im, lang, eval=True)
        sr_predictions.append(str_filt(rec, voc_types[lang]).replace(" ", ""))
        gt_str = str_filt(gt_strs[cnt].lower(), voc_types[all_langs[cnt]]).replace(" ", "") # all to be lower
        gt_strs_filt.append(gt_str)
        print(str_filt(rec, voc_types[lang]), "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
        cnt += 1
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt

@METRIC_REGISTRY.register()
def calculate_recognition_case_insensitive_langdivide(img, img2, crop_border, gt_lines, input_order='HWC', test_y_channel=False, **kwargs):

    polys, gt_strs = get_polys(gt_lines)
    sr_warpped = crop_images(img, polys)

    voc_types = {
        "ENG": "upper",
        "CHN": "benchmark"
    }

    sr_predictions = {
        "ENG": [],
        "CHN": []
    }

    gt_strs_filt = {
        "ENG": [],
        "CHN": []
    }

    cnt = 0
    all_langs = []
    for im in sr_warpped:
        lang = get_lang(gt_strs[cnt])
        all_langs.append(lang)
        rec = recognition_processing(im, lang, eval=True)
        pred_str = str_filt(rec, voc_types[lang]).replace(" ", "")
        sr_predictions[lang].append(pred_str)
        gt_str = str_filt(gt_strs[cnt].lower(), voc_types[all_langs[cnt]]).replace(" ", "") # all to be lower
        gt_strs_filt[lang].append(gt_str)
        print(pred_str, "||", gt_str)
        # cv2.imwrite(os.path.join(debug_visdir, "crop_img_" + gt_strs[cnt] + "_" + str_filt(rec) + ".jpg"), im)
        cnt += 1
    # [str_filt(gt_strs[idx], voc_types[all_langs[idx]]) for idx in range(len(gt_strs))]
    return sr_predictions, gt_strs_filt