import os
import cv2
import numpy as np


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


def get_TFSR(mode, gt_folder, scale):
    phase = mode
    gt_folder = os.path.join(gt_folder, phase) #"opt['phase']"

    fl_13mmdir = os.path.join(gt_folder, "13mm")
    fl_26mmdir = os.path.join(gt_folder, "26mm")
    fl_52mmdir = os.path.join(gt_folder, "52mm")

    paths = []

    # scale = opt['scale']

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
            paths.append(
                {"path_1x": impath_13mm, "path_2x": impath_26mm, "path_4x": impath_52mm, "gt_annos": anno_lines, 'syn_degr': False})

        else:
            anno_path = os.path.join(gt_folder, "trans_annos_52mm_renov", "res_" + imname.split(".")[0] + ".txt")
            anno_lines = open(anno_path, "r", encoding="utf-8").readlines()
            # print("anno_lines:", anno_lines)
            #if scale == 4:
            paths.append(
                {"path_1x": impath_13mm, "path_2x": impath_26mm, "path_4x": impath_52mm, "gt_annos": anno_lines,
                 'syn_degr': False})

    return paths




if __name__ == "__main__":

    root_dir = "/data0_ssd2t/majianqi/TFSR/"
    mode = "val"
    annos = get_TFSR(mode, root_dir, 4)

    cropped_root = "/data0_ssd2t/majianqi/TFSR_cropped/"

    for anno in annos[:]:
        imp_13mm = anno["path_1x"]
        imp_26mm = anno["path_2x"]
        imp_52mm = anno["path_4x"]

        imname = imp_13mm.split("/")[-1].split(".")[0]

        im_13mm = cv2.imread(imp_13mm)
        im_26mm = cv2.imread(imp_26mm)
        im_52mm = cv2.imread(imp_52mm)

        print("im:", im_13mm.shape, im_26mm.shape, im_52mm.shape)

        anno_lines = anno["gt_annos"]
        transcripts = [line.split(",")[-1].replace("\n", "") for line in anno_lines]
        polys = [np.array(line.split(",")[:8]).astype(np.uint).reshape(-1, 4, 2) for line in anno_lines]
        print("polys:", len(anno_lines))
        if len(anno_lines) > 0:
            polys = np.concatenate(polys, axis=0)

            cropped_ims_13mm = crop_images(im_13mm, polys)
            cropped_ims_26mm = crop_images(im_26mm, polys)
            cropped_ims_52mm = crop_images(im_52mm, polys)

            print("13mm:", len(cropped_ims_13mm), cropped_ims_13mm[0].shape)

            cropped_13mm_dir = os.path.join(cropped_root, "13mm")
            cropped_26mm_dir = os.path.join(cropped_root, "26mm")
            cropped_52mm_dir = os.path.join(cropped_root, "52mm")

            if not os.path.isdir(cropped_13mm_dir):
                os.makedirs(cropped_13mm_dir)
            if not os.path.isdir(cropped_26mm_dir):
                os.makedirs(cropped_26mm_dir)
            if not os.path.isdir(cropped_52mm_dir):
                os.makedirs(cropped_52mm_dir)

            for idx in range(len(cropped_ims_13mm)):
                cropped_13mm = cropped_ims_13mm[idx]
                cropped_26mm = cropped_ims_26mm[idx]
                cropped_52mm = cropped_ims_52mm[idx]

                transcript = transcripts[idx]

                cropped_13mm_impath = os.path.join(cropped_13mm_dir, imname + "_" + str(idx) + ".png")
                cropped_26mm_impath = os.path.join(cropped_26mm_dir, imname + "_" + str(idx) + ".png")
                cropped_52mm_impath = os.path.join(cropped_52mm_dir, imname + "_" + str(idx) + ".png")

                cv2.imwrite(cropped_13mm_impath, cropped_13mm)
                cv2.imwrite(cropped_26mm_impath, cropped_26mm)
                cv2.imwrite(cropped_52mm_impath, cropped_52mm)

                print("image:", cropped_13mm_impath, anno_lines[idx])