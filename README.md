# [Real-CE: A Benchmark for Chinese-English Scene Text Image Super-resolution (ICCV2023)](https://arxiv.org/abs/2308.03262)

_[Jianqi Ma](https://scholar.google.com/citations?user=kQUJjQQAAAAJ&hl=en), Zhetong Liang, Wangmeng Xiang, Xi Yang, [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang)_  
_Department of Computing, [The Hong Kong Polytechnic University](http://www.comp.polyu.edu.hk), Hong Kong, China & OPPO Research_

## License and Agreement
1. Please note that this dataset is made available for **non-commercial** academic research purposes **ONLY**. 

2. You agree not to reproduce, duplicate, copy, sell, trade, resell or exploit for any commercial purposes, any portion of the images and any portion of derived data.

3. You agree not to further copy, publish or distribute any portion of the RealCE dataset. Except, for internal use at a single site within the same organization it is allowed to make copies of the dataset.

4. The image contents are released upon request for research purposes only. Please contact us for details.

5. Any violation of this protocol will be at his own risk. If any of the images include your information and you would like to remove them, please kindly inform us, and we will remove them from our dataset immediately.

## Dataset Access
Coming soon...

## Benchmark Installation

This project is mainly built upon [BasicSR](https://github.com/XPixelGroup/BasicSR). Please see `BASICSR_INSTALLATION.md` or the original repo for installation details. The recognition evaluation needs Chinese and English recognizer weights. They can be downloaded from the link: https://drive.google.com/file/d/1wga0xFdBSkAt_Pif3wPMG4tnHA9wQ7wD/view?usp=sharing

After downloading the pth files, you need to place the files in `<ROOT>/basicsr/metrics/` for weights loading.

### Run the evaluation for RRDB

```
bash test_RRDB_RealCEx4.sh
```

### Run the training code for RRDB

```
bash train_RRDB_RealCEx4.sh
```

