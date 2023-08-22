PYTHONPATH="./:${PYTHONPATH}" \
	CUDA_VISIBLE_DEVICES=6,7 \
	python basicsr/train.py -opt options/train/ELAN/train_ELAN_SRx4_scratch_TFSR_inoutCanny.yml
