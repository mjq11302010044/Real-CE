import os

data_dir = "/data0_ssd2t/majianqi/data/"
tar_dir = "/data0_ssd2t/majianqi/DocSR_Split"

tar_lr_dir = os.path.join(tar_dir, "blur")
tar_hr_dir = os.path.join(tar_dir, "orig")

if not os.path.isdir(tar_hr_dir):
    os.makedirs(tar_hr_dir)

if not os.path.isdir(tar_lr_dir):
    os.makedirs(tar_lr_dir)


imlist = os.listdir(data_dir)

imkeys = {}

for imname in imlist:
    if "png" in imname:
        imkey = imname.split("_")[0]
        if not imkey in imkeys:
            imkeys[imkey] = 1
cnt = 0
all_num = len(imkeys.keys())
for key in imkeys:
    lr_name = key + "_blur.png"
    hr_name = key + "_orig.png"

    cmd_lr = "cp " + os.path.join(data_dir, lr_name) + " " + os.path.join(tar_lr_dir, key + ".png")
    cmd_hr = "cp " + os.path.join(data_dir, hr_name) + " " + os.path.join(tar_hr_dir, key + ".png")

    os.popen(cmd_lr)
    os.popen(cmd_hr)
    cnt += 1
    if cnt % 1000 == 0:
        print(cnt, "/", all_num)