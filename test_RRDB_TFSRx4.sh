PYTHONPATH="./:${PYTHONPATH}" \
	CUDA_VISIBLE_DEVICES=5 \
	python basicsr/test.py -opt options/test/ESRGAN/test_RRDBNet_PSNR_x4_TFSR_wCanny_MaskedCannyperceptual_deformation.yml
