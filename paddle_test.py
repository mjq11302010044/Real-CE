import os
import cv2
from paddleocr import PaddleOCR

impath = "/data1_ssd4t/majianqi/experiments/002_GlobalRRDBNet_x2_TFSR_RGB_Canvas_CannyPerceptual_withHQSynth_GlobalCanny_TE_STISRv2/visualization/ip11pro_output_TextSR_renew_88/ip11pro_output_TextSR_renew_88_150000.png"

# image = cv2.imread(impath)

ocr_parser = PaddleOCR(use_angle_cls=True, lang="ch")
ocr_res = ocr_parser.ocr(impath, cls=True)

print("------------- OCR Results -------------")
for line in ocr_res:
    print(line)