import os
import cv2

models_map = {
        "RRDB": {
        "L1":"~/workspace/BasicSR/experiments/051_RRDBNet_PSNR_x4_TFSR_RealSR/",
        "EA_P":"/data1_ssd4t/majianqi/experiments/train_RRDB_SRx2_scratch_v2",
        # "EA_ALL":"/home/majianqi/workspace/BasicSR/experiments/051_RRDBNet_PSNR_x4_TFSR_ft_wCanny_Cannyperceptual_v2"
        },
        "EDSR": {
        "L1":"/data1_ssd4t/majianqi/experiments/203_EDSR_Mx4_TextZoom",
        "EA_P":"/data1_ssd4t/majianqi/experiments/203_EDSR_Mx4_RealSR",
        # "EA_ALL":"/data1_ssd4t/majianqi/experiments/203_EDSR_Mx4_TFSR_ft_wCanny_Cannyperceptual"
        },
        "SRResNet":{
        "L1":"/data1_ssd4t/majianqi/experiments/203_SRResNet_Mx4_TextZoom/",
        "EA_P":"/data1_ssd4t/majianqi/experiments/203_SRResNet_Mx4_RealSR/",
        # "EA_ALL":"/home/majianqi/workspace/BasicSR/experiments/001_MSRResNet_TFSR_ft_MaskedL1_wCanny_perceptual"
        },
        "ELAN":{
        # "L1":"/data1_ssd4t/majianqi/experiments/train_ELAN_SRx4_TextZoom",
        "EA_P":"/data1_ssd4t/majianqi/experiments/train_ELAN_SRx4_RealSR",
        # "EA_ALL":"/data1_ssd4t/majianqi/experiments/train_ELAN_SRx4_scratch_ln_canny_cannyperceptual_continue_v2"
        },
        "RCAN":{
        # "L1":"/data1_ssd4t/majianqi/experiments/201_RCANx4_scratch_TextZoom",
        # "EA_P":"/data1_ssd4t/majianqi/experiments/201_RCANx4_scratch_RealSR",
        # "EA_ALL":"/home/majianqi/workspace/BasicSR/results/201_RCANx4_scratch_TFSR_ft_wCanny_Cannyperceptual_test"
        }}

models_map_x2 = {
        "RRDB": {
        "L1":"/data1_ssd4t/majianqi/experiments/train_RRDB_SRx2_scratch_v2_archived_20221106_142137",
        "EA_P":"/data1_ssd4t/majianqi/experiments/train_RRDB_SRx2_scratch_v2_wCanny",
        "EA_ALL":"/data1_ssd4t/majianqi/experiments/train_RRDB_SRx2_scratch_v2_wCanny_perceptual"
        },
        "EDSR": {
        "L1":"/data1_ssd4t/majianqi/experiments/203_EDSR_Mx2_TFSR/",
        "EA_P":"/data1_ssd4t/majianqi/experiments/203_EDSR_Mx2_TFSR_wCanny/",
        "EA_ALL":"/data1_ssd4t/majianqi/experiments/203_EDSR_Mx2_TFSR_wCanny_Cannyperceptual/"
        },
        "SRResNet":{
        "L1":"~",
        "EA_P":"~",
        "EA_ALL":"~"
        },
        "ELAN":{
        "L1":"/data1_ssd4t/majianqi/experiments/train_ELAN_SRx4_ln_version/",
        "EA_P":"/data1_ssd4t/majianqi/experiments/train_ELAN_SRx2_scratch_ln_wcanny/",
        "EA_ALL":"/data1_ssd4t/majianqi/experiments/train_ELAN_SRx4_scratch_ln_canny_cannyperceptual_continue_v2/"
        },
        "RCAN":{
        "L1":"/data1_ssd4t/majianqi/experiments/201_RCANx2_scratch_TFSR",
        "EA_P":"/data1_ssd4t/majianqi/experiments/201_RCANx2_scratch_TFSR_ft_wCanny",
        "EA_ALL":"/data1_ssd4t/majianqi/experiments/201_RCANx2_scratch_TFSR_wCanny_perceptual/"
        }}


iters = {
    "L1": 20000,
    "EA_P": 20000,
    "EA_ALL": 40000
}

def imname_check(imname_need, imlist):
    for im in imlist:
        if imname_need in im:
            return im


image_ids = ['ip11pro_output_textSR_211206_renew_101', 'ip12pro_output_103APPLE_renew_23',
             'ip12pro_output_102APPLE_renew_102', 'ip12pro_output_103APPLE_renew_2',
             'ip11pro_output_testSR_2101225_xwm_renew_46', 'ip11pro_output_textSR_211206_renew_30',
             'ip11pro_output_textSR_211206_renew_59', 'ip11pro_output_textSR_211206_renew_49',
             'ip11pro_output_TextSR_renew_48', 'ip11pro_output_TextSR-20211222_renew_3',
             'ip11pro_output_xwm_testSR_211211_renew_4']

#models = models_map.keys()

tar_dir = "target_picked_dir_RealSR"
if not os.path.isdir(tar_dir):
    os.makedirs(tar_dir)

for model_name in models_map:

    for loss_term in models_map[model_name]:
        imdir = models_map[model_name][loss_term]

        if "experiments" in imdir:
            for id in image_ids:
                impath = os.path.join(imdir, "visualization", id, id + "_" + str(iters[loss_term]) + ".png")
                tar_path = os.path.join(tar_dir, model_name + "_" + loss_term + "_" + id + ".png")
                cmd = "cp " + impath + " " + tar_path
                res = os.popen(cmd).readlines()
                print("cmd:", cmd, res)
        else:
            imlist = os.listdir(os.path.join(imdir, "visualization", "13mm"))

            for id in image_ids:
                imname = imname_check(id, imlist)
                impath = os.path.join(imdir, "visualization", "13mm", imname)
                tar_path = os.path.join(tar_dir, model_name + "_" + loss_term + "_" + id + ".png")
                cmd = "cp " + impath + " " + tar_path
                res = os.popen(cmd).readlines()
                print("cmd:", cmd, res)


