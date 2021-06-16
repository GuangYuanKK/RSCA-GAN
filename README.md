# RSCA-GAN

Environment setting: cuda 8.0 + cudnn 6.0 + tensorflow 1.4 + tensorpack 0.8

Train parameters: python train_RSCA.py --gpu=' ' --imageDir=' ' --labelDir=' ' --maskDir=' ' 

Test parameters: python train_RSCA.py --gpu=' ' --imageDir=' ' --labelDir=' ' --maskDir=' ' --sample='result/knee_mask_8x/' --load='train_log/knee_mask_8x/max-validation_PSNR_boost_A.data-00000-of-00001'
