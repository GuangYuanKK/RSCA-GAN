# RSCA-GAN

Environment setting: cuda 8.0 + cudnn 6.0 + tensorflow 1.4 + tensorpack 0.8

Train parameters: python train_RSCA.py --gpu=' ' --imageDir=' ' --labelDir=' ' --maskDir=' ' 

Test parameters: python train_RSCA.py --gpu=' ' --imageDir=' ' --labelDir=' ' --maskDir=' ' --sample='result/knee_mask_8x/' --load='train_log/knee_mask_8x/max-validation_PSNR_boost_A.data-00000-of-00001'

Paper: 

A Modified Generative Adversarial Network Using Spatial and Channel-Wise Attention for CS-MRI Reconstruction https://ieeexplore.ieee.org/abstract/document/9447721

Cite This:

@ARTICLE{9447721,  
author={Li, Guangyuan and Lv, Jun and Wang, Chengyan},  
journal={IEEE Access},   
title={A Modified Generative Adversarial Network Using Spatial and Channel-Wise Attention for CS-MRI Reconstruction},   
year={2021},  
volume={9},  
number={},  
pages={83185-83198},  
doi={10.1109/ACCESS.2021.3086839}}
