import os
import random

ori_root = "/data0_ssd2t/majianqi/output_TR_SR/"

device_list = os.listdir(ori_root)

focal_dict = {
                "1x": "13mm",
                "2x": "26mm",
                "4x": "52mm",
              }

imkey_set = {}

for device in device_list:

    print("device:", device)
    device_dir = os.path.join(ori_root, device)
    sublist = os.listdir(device_dir)

    for subname in sublist:

        imdir = os.path.join(device_dir, subname)
        imlist = os.listdir(imdir)

        for imname in imlist:
            prefix_key, _, sr = imname.split(".")[0].split("_")
            focal_len = focal_dict[sr]
            imkey = device + "_" + subname + "_" + prefix_key

            if not imkey in imkey_set:
                imkey_set[imkey] = {focal_len:os.path.join(imdir, imname)}
            else:
                imkey_set[imkey][focal_len] = os.path.join(imdir, imname)

            # print("imkey:", imkey)



# print(imkey_set)

imkey_list = list(imkey_set.keys())
random.shuffle(imkey_list)

# print(len(imkey_list))

imkeylist_train = imkey_list[300:]
imkeylist_test = imkey_list[:300]

tar_dir = "/data0_ssd2t/majianqi/TFSR/"
tar_dir_train = "/data0_ssd2t/majianqi/TFSR/train"
tar_dir_test = "/data0_ssd2t/majianqi/TFSR/test"

if not os.path.isdir(tar_dir_train):
    os.makedirs(tar_dir_train)

if not os.path.isdir(tar_dir_test):
    os.makedirs(tar_dir_test)

def split_sets(key_list, split_dir):

    for k in key_list:

        impath_set = imkey_set[k]

        for focal_len in impath_set:

            if not os.path.isdir(os.path.join(split_dir, focal_len)):
                os.makedirs(os.path.join(split_dir, focal_len))

            cmd = "cp " + impath_set[focal_len] + " " + os.path.join(split_dir, focal_len, k + ".JPG")

            print(os.popen(cmd))



if __name__ == "__main__":
    split_sets(imkeylist_train, tar_dir_train)
    split_sets(imkeylist_test, tar_dir_test)