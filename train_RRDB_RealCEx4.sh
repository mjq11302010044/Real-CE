PYTHONPATH="./:${PYTHONPATH}" \
	CUDA_VISIBLE_DEVICES=4 \
	python basicsr/train.py -opt options/train/ESRGAN/train_RRDBNet_PSNR_x4_RealCE_wCanny_Cannyperceptual.yml
