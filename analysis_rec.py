import os
import editdistance


global_res_dir = "/home/majianqi/workspace/BasicSR/results/test_RRDB_SRx2_scratch_v2_wCanny_Cannyperceptual_retest/visualization/TFSR"
local_res_dir = "/data1_ssd4t/majianqi/experiments/002_GlobalRRDBNet_x2_TFSR_RGB_Canvas_CannyPerceptual_withHQSynth_GlobalCanny_TE_STISRv3_discrete/visualization/"

global_f_list = os.listdir(global_res_dir)

all_dict = {}

global_rec_list = [filename for filename in global_f_list if filename.split(".")[-1] == "txt"]

local_dir_list = os.listdir(local_res_dir)

for imdir in local_dir_list:
    all_dict[imdir] = {"local_res": imdir + "/" + imdir + "_60000.txt", "global_res": None}

for fname in global_rec_list:
    obj_name = fname.replace("_Cannyperceptual.txt", "")
    all_dict[obj_name]["global_res"] = fname


for key in all_dict:
    local_res_file = os.path.join(local_res_dir, all_dict[key]["local_res"])
    global_res_file = os.path.join(global_res_dir, all_dict[key]["global_res"])

    local_lines = open(local_res_file, 'r').readlines()
    global_lines = open(global_res_file, 'r').readlines()

    # print("key:", key)

    for i in range(len(local_lines)):

        l_line, g_line = local_lines[i], global_lines[i]
        l_pred = l_line.replace("\n", "").split("\t")[0]
        g_pred = g_line.replace("\n", "").split("\t")[0]
        gt = l_line.replace("\n", "").split("\t")[1]

        gt_len = len(gt)
        l_dis = 1 - editdistance.eval(l_pred, gt) / max(len(l_pred), gt_len)
        g_dis = 1 - editdistance.eval(g_pred, gt) / max(len(g_pred), gt_len)
        print("lines:", l_dis, "|", g_dis, "|", l_pred, "|", g_pred, "|", gt, "|", key)