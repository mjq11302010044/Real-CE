import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#max_height: 4032.0
#max_width: 3024.0
#min_height: 200.0
#min_width: 396.0
#cnt_line_train: 7849
#cnt_line_test: 3414
#cnt_line_chn: 21436
#cnt_line_eng: 8756
#line_nums: 11263
#max_line_height: 1156.0
#max_line_width: 2883.0
#min_line_height: 1.0
#min_line_width: 2.0
#text_length: {'x<2': 71, '2≤x<5': 1107, '5≤x<8': 749, 'x≥8': 9336}
#ratios: {'x<2': 1778, '2≤x<5': 3811, '5≤x<8': 2369, 'x≥8': 3305}
#line_height: {'x<32': 1664, '32≤x<64': 5211, 'x≥64': 4388}
#train_global_image: 576
#test_global_image: 261

root_dir = "/data0_ssd2t/majianqi/TFSR"

def is_chinese(ipath):
    for i in range(len(ipath)):
        if ipath[i] > u'\u4e00' and ipath[i] < u'\u9fff':
            return True

    return False

def cnt_chinese(ipath):
    cnt = 0
    for i in range(len(ipath)):
        if ipath[i] > u'\u4e00' and ipath[i] < u'\u9fff':
            cnt += 1

    return cnt

def crop_images(global_image, polys):
    # img = cv2.imread("big_vertical_text.jpg")
    # points for test.jpg
    all_warpped = []
    # print("polys:", polys)
    rboxes = []
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
            rect = (rect[0], (height, width), (90 - abs(rect[2]))) ############
        width = int(rect[1][0])
        height = int(rect[1][1])

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        rboxes.append([min(height, width), max(height, width)])

    return rboxes

polys = []
transcripts = []
im_shape = []
cnt_line_train = 0
cnt_line_test = 0
cnt_line_chn = 0
cnt_line_eng = 0

for split in ["val"]:

    anno_d = "trans_annos_52mm" if split == "train" else "annos"

    im_dir = "/data0_ssd2t/majianqi/TFSR/" + split + "/52mm"
    anno_dir = "/data0_ssd2t/majianqi/TFSR/" + split + "/" + anno_d

    valid_file = "/data0_ssd2t/majianqi/TFSR/" + split + "/valid_list.txt"

    valid_list = open(valid_file, "r").readlines()

    imlist = [imname.replace("\n", "") for imname in valid_list]

    # imlist = os.listdir(im_dir)

    for imname in imlist:
        impath = os.path.join(im_dir, imname)
        if split == "train":
            annopath = os.path.join(anno_dir, "res_" + imname.split(".")[0] + ".txt")
        else:
            annopath = os.path.join(anno_dir, imname.split(".")[0] + ".txt")

        image = cv2.imread(impath)

        H, W = image.shape[:2]
        print("annopath:", annopath)
        if split == "train":
            anno_lines = open(annopath, "r", encoding="utf-8").readlines()
        else:
            anno_lines = open(annopath, "r", encoding="gbk").readlines()
        im_shape.append([H, W])

        for anno_line in anno_lines:
            items = anno_line.split(",")
            poly = np.array(items[:8]).astype(np.float32)
            trans = items[8].replace("\n", "")
            polys.append(poly)
            transcripts.append(trans)
            if split == "train":
                cnt_line_train += 1
            else:
                cnt_line_test += 1

            if is_chinese(trans):
                cnt_line_chn += cnt_chinese(trans)
            else:
                cnt_line_eng += 1

rboxes = crop_images(image, np.array(polys).reshape(-1, 4, 2))

im_shape = np.array(im_shape).astype(np.float32)
rboxes = np.array(rboxes).astype(np.float32)

max_height = np.max(im_shape[:, 0])
max_width = np.max(im_shape[:, 1])
min_height = np.min(im_shape[:, 0])
min_width = np.min(im_shape[:, 1])

print("max_height:", max_height)
print("max_width:", max_width)
print("min_height:", min_height)
print("min_width:", min_width)
print("cnt_line_train:", cnt_line_train)
print("cnt_line_test:", cnt_line_test)
print("cnt_line_chn:", cnt_line_chn)
print("cnt_line_eng:", cnt_line_eng)

max_len = 0
min_len = 999
for trans in transcripts:
    if len(trans) > max_len:
        max_len = len(trans)
    if len(trans) < min_len:
        min_len = len(trans)

max_line_height = np.max(rboxes[:, 0])
max_line_width = np.max(rboxes[:, 1])
min_line_height = np.min(rboxes[:, 0])
min_line_width = np.min(rboxes[:, 1])

print("line_nums:", len(polys))
print("max_line_height:", max_line_height)
print("max_line_width:", max_line_width)
print("min_line_height:", min_line_height)
print("min_line_width:", min_line_width)

# line height: [32, 64]
# text length : [3, 5, 8]
# ratio: [3, 5, 8]

line_height = {"x<32":0, "32≤x<64":0, "x≥64":0}
text_length = {"x<2":0, "2≤x<5":0, "5≤x<8":0, "x≥8":0}
ratios = {"x<2":0, "2≤x<5":0, "5≤x<8":0, "x≥8":0}

cnt = 0
for box in rboxes:

    trans = transcripts[cnt]
    len_tran = len(trans)
    height = rboxes[cnt][0]
    ratio = rboxes[cnt][1] / (float(rboxes[cnt][0]) + 1e-10)

    if len_tran < 2:
        text_length["x<2"] += 1
    elif 2 <= len_tran < 5:
        text_length["2≤x<5"] += 1
    elif 5 <= len_tran < 8:
        text_length["5≤x<8"] += 1
    elif len_tran >= 8:
        text_length["x≥8"] += 1

    if ratio < 2:
        ratios["x<2"] += 1
    elif 2 <= ratio < 5:
        ratios["2≤x<5"] += 1
    elif 5 <= ratio < 8:
        ratios["5≤x<8"] += 1
    elif ratio >= 8:
        ratios["x≥8"] += 1

    if height < 32:
        line_height["x<32"] += 1
    elif 32 <= height < 64:
        line_height["32≤x<64"] += 1
    elif height >= 64:
        line_height["x≥64"] += 1

    cnt += 1

print("text_length:", text_length)
print("ratios:", ratios)
print("line_height:", line_height)
