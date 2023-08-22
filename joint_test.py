import os
import sys

opt_root = "./options/test/"

model_prefix = {
    # "RRDB": "ESRGAN/test_RRDBNet_",
    "ELAN": "ELAN/test_ELAN_SR",
    # "EDSR": "EDSR/test_EDSR_M",
    # "RCAN": "RCAN/test_RCAN",
    "SRResNet": "SRResNet_SRGAN/test_MSRResNet_"
}

factors = ["x4", "x2"]
settings = ["TFSR", "TFSR_wCanny", "TFSR_wCanny_Cannyperceptual"]

all_configs = []

for fact in factors:
    for model in model_prefix:
        for set_item in settings:

            if model == "ELAN":
                option_file = opt_root + model_prefix[model] + fact + "_scratch_" + set_item + ".yml"
            else:
                option_file = opt_root + model_prefix[model] + fact + "_" + set_item + ".yml"

            all_configs.append(option_file)

all_configs = [
    './options/test/ELAN/test_ELAN_SRx2_scratch_TFSR.yml',
    # './options/test/ELAN/test_ELAN_SRx2_scratch_TFSR_wCanny.yml',
    './options/test/ELAN/test_ELAN_SRx2_scratch_TFSR_wCanny_Cannyperceptual.yml',
]

for config in all_configs:
        cmd = 'PYTHONPATH="./:{$PYTHONPATH}" CUDA_VISIBLE_DEVICES=5 python basicsr/test.py -opt ' + config
        print("cmd:", cmd)
        content = os.popen(cmd).read()
        print("All content:", content)
