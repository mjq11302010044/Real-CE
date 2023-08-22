PYTHONPATH="./:${PYTHONPATH}" \
	CUDA_VISIBLE_DEVICES=4 \
	python basicsr/train.py -opt options/train/ESRGAN/train_RRDBNet_PSNR_x4_TFSR_wCanny_MaskedCannyperceptual_deformation.yml
