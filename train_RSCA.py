from RSCA_GAN import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
if __name__ == '__main__':
	main()