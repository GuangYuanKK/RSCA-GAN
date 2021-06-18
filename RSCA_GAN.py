from tensorpack.tfutils import symbolic_functions

from Utils_SCA import *

import os, sys
import argparse
import glob
from six.moves import map, zip, range
import numpy as np

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack.tfutils.symbolic_functions as symbf
import tensorflow as tf
from GAN import GANTrainer, GANModelDesc, SeparateGANTrainer, MultiGPUGANTrainer

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2

class Model(GANModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'inputA'),
				InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'mask'), 
				InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'inputB'), 
				]

	def build_losses(self, vecpos, vecneg, name="WGAN_loss"):
		with tf.name_scope(name=name):
			# the Wasserstein-GAN losses
			d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
			g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')#tf.negative计算数值的负值元素,即：y = -x
			# add_moving_summary(self.d_loss, self.g_loss)
			return g_loss, d_loss

	# def build_losses(self, real, fake, name="LSGAN_loss"):
	# 	d_real = tf.reduce_mean(tf.squared_difference(real, 1), name='d_real')
	# 	d_fake = tf.reduce_mean(tf.square(fake), name='d_fake')
	# 	d_loss = tf.multiply(d_real + d_fake, 0.5, name='d_loss')
	# 	tf.summary.histogram('score-real', d_real)
	# 	tf.summary.histogram('score-fake', d_fake)
	# 	g_loss = tf.reduce_mean(tf.squared_difference(fake, 1), name='g_loss')
	# 	# add_moving_summary(g_loss, d_loss)
	# 	return g_loss, d_loss


	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img):
		assert img is not None
		return arch_generator(img)
		# return arch_fusionnet(img)

	@auto_reuse_variable_scope
	def discriminator(self, img):
		assert img is not None
		return arch_discriminator(img)

	@auto_reuse_variable_scope
	# def expansion(self, img):
	# 	assert img is not None
	# 	return arch_fusionnet(img)
	


	
	def _build_graph(self, inputs):
		A, R, B = inputs
		# A, R, B, C = inputs

		A = cvt2tanh(A)
		R = cvt2tanh(R)
		B = cvt2tanh(B)
		#------------------------------lgy
		# C = cvt2tanh(C)

		A = tf.identity(A, name='A')
		R = tf.identity(R, name='R')
		B = tf.identity(B, name='B')
		#-------------------------------lgy
		# C = tf.identity(C, name='C')



		# use the initializers from torch
		#from tensorflow.python.keras._impl.keras.layers import LeakyReLU
		#from tensorflow.python.keras._impl.keras.backend import relu
		with argscope([Conv2D, Deconv2D, FullyConnected],
					  # W_init=tf.contrib.layers.variance_scaling_initializer(factor=.333, uniform=True),
					  W_init=tf.truncated_normal_initializer(stddev=0.02),
					  #tf.truncated_normal_initializer的意思是：从截断的正态分布中输出随机值
					  #生成的值服从具有指定平均值和标准偏差的正态分布，如果生成的值大于平均值2个标准偏差的值则丢弃重新选择
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NCHW'), \
				argscope(PReLU, alpha=0.2):
			with tf.name_scope('preprocessing'):
				S01  = tf.identity(A, name='S01') # For PSNR
				S02  = tf.identity(B, name='S02') # For PSNR
				#----------------------------------lgy
				# S03  = tf.identity(C, name='S03')

				R  = tf.identity(R, name='R')
				Rh = tf.conj(R, name='Rh')
				#tf.conj(input, name=None) 	计算共轭复数

				Mf01 = RF(S01, R)
				Mf02 = RF(S02, R)
				#------------------------------------lgy
				# Mf03 = RF(S03, R)

				# Zero filling
				M1   = FhRh(Mf01, R, name='M1')
				M2   = FhRh(Mf02, R, name='M2')
				#------------------------------------lgy
				# M3   = FhRh(Mf03, R, name='M3')
					
			with tf.variable_scope('gen'):
				with tf.variable_scope('recon'):
					Sn1  = self.generator(M1)
					Sn2  = self.generator(M2)
					#-------------------------------lgy
					# Sn3  = self.generator(M3)

			Sp1  = Sn1+M1
			S1   = update(Sp1, S01, R, name='S1')
			
			Sp2  = Sn2+M2
			S2   = update(Sp2, S02, R, name='S2')
			
			with tf.variable_scope('gen'):
				with tf.variable_scope('boost'):
					Tn1  = self.generator(S1)
					Tn2  = self.generator(S2)

			Tp1  = Tn1+S1
			T1   = update(Tp1, S01, R, name='T1')
			Tp2  = Tn2+S2
			T2   = update(Tp2, S02, R, name='T2')

			#-------------------------------------lgy begin
			#with tf.variable_scope('gen'):
			#	with tf.variable_scope('boost-lgy'):
			#		Yn1  = self.generator(Tn1)
			#		Yn2  = self.generator(Tn2)

			#Yp1  = Yn1+T1
			#Y1   = update(Yp1, S01, R, name='Y1')
			#Yp2  = Yn2+T2
			#Y2   = update(Yp2, S02, R, name='Y2')
			#--------------------------------------lgy end



			
			with tf.variable_scope('discrim'):
				S1_dis_real = self.discriminator(S01)
				S1_dis_fake = self.discriminator(S1)
				T1_dis_fake = self.discriminator(T1)
				#----------------------------------------lgy
				#Y1_dis_fake = self.discriminator(Y1)
				
				S2_dis_real = self.discriminator(S02)
				S2_dis_fake = self.discriminator(S2)
				T2_dis_fake = self.discriminator(T2)
				#----------------------------------------lgy
				#Y2_dis_fake = self.discriminator(Y2)


		with tf.name_scope('losses'):
			with tf.name_scope('Frq'):
				with tf.name_scope('Recon'):
					recon_frq_AA = tf.reduce_mean(tf.abs((RF(S01, R) - RF(Sp1, R))), name='recon_frq_AA')
					# tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
					recon_frq_BB = tf.reduce_mean(tf.abs((RF(S02, R) - RF(Sp2, R))), name='recon_frq_BB')
				with tf.name_scope('Boost'):
					recon_frq_Aa = tf.reduce_mean(tf.abs((RF(S01, R) - RF(Tp1, R))), name='recon_frq_Aa')
					recon_frq_Bb = tf.reduce_mean(tf.abs((RF(S02, R) - RF(Tp2, R))), name='recon_frq_Bb')
				#-------------------------------------lgy
				#with tf.name_scope('Boost-lgy'):
				#	recon_frq_aa = tf.reduce_mean(tf.abs((RF(S01, R) - RF(Yp1, R))), name='recon_frq_aa')
				#	recon_frq_bb = tf.reduce_mean(tf.abs((RF(S02, R) - RF(Yp2, R))), name='recon_frq_bb')

			with tf.name_scope('Img'):
				with tf.name_scope('Zfill'):
					zfill_img_MA = tf.reduce_mean(tf.abs((S01) - (M1)), name='zfill_img_MA')
					zfill_img_MB = tf.reduce_mean(tf.abs((S02) - (M2)), name='zfill_img_MB')
					
				with tf.name_scope('Recon'):
					recon_img_AA = tf.reduce_mean(tf.abs((S01) - (S1)), name='recon_img_AA')
					recon_img_BB = tf.reduce_mean(tf.abs((S02) - (S2)), name='recon_img_BB')
					error_img_AA = tf.reduce_mean(tf.abs((S01) - (Sp1)), name='error_img_AA')
					error_img_BB = tf.reduce_mean(tf.abs((S02) - (Sp2)), name='error_img_BB')
					smoothness_AA = tf.reduce_mean(tf.image.total_variation((S1)), name='smoothness_AA')
					smoothness_BB = tf.reduce_mean(tf.image.total_variation((S2)), name='smoothness_BB')
			
				with tf.name_scope('Boost'):
					recon_img_Aa = tf.reduce_mean(tf.abs((S01) - (T1)), name='recon_img_Aa')
					recon_img_Bb = tf.reduce_mean(tf.abs((S02) - (T2)), name='recon_img_Bb')
					error_img_Aa = tf.reduce_mean(tf.abs((S01) - (Tp1)), name='error_img_Aa')
					error_img_Bb = tf.reduce_mean(tf.abs((S02) - (Tp2)), name='error_img_Bb')
					smoothness_Aa = tf.reduce_mean(tf.image.total_variation((T1)), name='smoothness_Aa')
					smoothness_Bb = tf.reduce_mean(tf.image.total_variation((T2)), name='smoothness_Bb')
				#-----------------------------------------lgy begin
				#with tf.name_scope('Boost-lgy'):
				#	recon_img_aa = tf.reduce_mean(tf.abs((S01) - (Y1)), name='recon_img_aa')
				#	recon_img_bb = tf.reduce_mean(tf.abs((S02) - (Y2)), name='recon_img_bb')
				#	error_img_aa = tf.reduce_mean(tf.abs((S01) - (Yp1)), name='error_img_aa')
				#	error_img_bb = tf.reduce_mean(tf.abs((S02) - (Yp2)), name='error_img_bb')
				#	smoothness_aa = tf.reduce_mean(tf.image.total_variation((Y1)), name='smoothness_aa')
				#	smoothness_bb = tf.reduce_mean(tf.image.total_variation((Y2)), name='smoothness_bb')
				#-----------------------------------------lgy end
				
			with tf.name_scope('LossAA'):
				G_loss_AA, D_loss_AA = self.build_losses(S1_dis_real, S1_dis_fake, name='AA')
				G_loss_Aa, D_loss_Aa = self.build_losses(S1_dis_real, T1_dis_fake, name='Aa')
				#--------------------------lgy
				#G_loss_aa, D_loss_aa = self.build_losses(S1_dis_real, Y1_dis_fake, name='aa')
			with tf.name_scope('LossBB'):
				G_loss_BB, D_loss_BB = self.build_losses(S2_dis_real, S2_dis_fake, name='BB')
				G_loss_Bb, D_loss_Bb = self.build_losses(S2_dis_real, T2_dis_fake, name='Bb')
				#--------------------------lgy
				#G_loss_bb, D_loss_bb = self.build_losses(S2_dis_real, Y2_dis_fake, name='bb')
			with tf.name_scope('LossAB'):
				G_loss_AB, D_loss_AB = self.build_losses(S1_dis_real, S2_dis_fake, name='AB')
				G_loss_Ab, D_loss_Ab = self.build_losses(S1_dis_real, T2_dis_fake, name='Ab')
				#--------------------------lgy
				#G_loss_ab, D_loss_ab = self.build_losses(S1_dis_real, Y2_dis_fake, name='ab')
			with tf.name_scope('LossBA'):
				G_loss_BA, D_loss_BA = self.build_losses(S2_dis_real, S1_dis_fake, name='BA')
				G_loss_Ba, D_loss_Ba = self.build_losses(S2_dis_real, T1_dis_fake, name='Ba')
				#--------------------------lgy
				#G_loss_ba, D_loss_ba = self.build_losses(S2_dis_real, Y1_dis_fake, name='ba')

		
		
		
						
		ALPHA = 1e+1 #10
		GAMMA = 1e-0 #1
		DELTA = 1e-4 #0.0001
		RATES = tf.count_nonzero(tf.ones_like(R), dtype=tf.float32) / 2 / tf.count_nonzero(R, dtype=tf.float32) 
		GAMMA = RATES 
		self.g_loss = tf.add_n([
								(G_loss_AA + G_loss_BB + G_loss_AB + G_loss_BA),
								(G_loss_Aa + G_loss_Bb + G_loss_Ab + G_loss_Ba),
								#----------------------------------------------lgy
								#(G_loss_aa + G_loss_bb + G_loss_ab + G_loss_ba),
								(recon_img_AA + recon_img_BB) * 1.00 * ALPHA * RATES, 
								(recon_img_Aa + recon_img_Bb) * 1.00 * ALPHA * RATES,
								#----------------------------------------------lgy
								#(recon_img_aa + recon_img_bb) * 1.00 * ALPHA * RATES,
								(error_img_AA + error_img_BB) * 1e+2 * ALPHA * RATES, 
								(error_img_Aa + error_img_Bb) * 1e+2 * ALPHA * RATES,
								# ----------------------------------------------lgy
								#(error_img_aa + error_img_bb) * 1e+2 * ALPHA * RATES,
								(recon_frq_AA + recon_frq_BB) * 1.00 * GAMMA * RATES, 
								(recon_frq_Aa + recon_frq_Bb) * 1.00 * GAMMA * RATES,
								#-----------------------------------------------lgy
								#(recon_frq_aa + recon_frq_bb) * 1.00 * GAMMA * RATES,
								(smoothness_AA + smoothness_BB + smoothness_Aa + smoothness_Bb) * DELTA,
								], name='G_loss_total')
		self.d_loss = tf.add_n([
								(D_loss_AA + D_loss_BB + D_loss_AB + D_loss_BA), 
								(D_loss_Aa + D_loss_Bb + D_loss_Ab + D_loss_Ba),
								#---------------------------------------------lgy
								#(D_loss_aa + D_loss_bb + D_loss_ab + D_loss_ba),
								], name='D_loss_total')

		wd_g = regularize_cost('gen/.*/W', 		l1_regularizer(1e-5), name='G_regularize')#regularize_cost正则化损失函数
		wd_d = regularize_cost('discrim/.*/W', 	l1_regularizer(1e-5), name='D_regularize')

		self.g_loss = tf.add(self.g_loss, wd_g, name='g_loss')
		self.d_loss = tf.add(self.d_loss, wd_d, name='d_loss')

	

		self.collect_variables()

		add_moving_summary(self.d_loss, self.g_loss)#add_moving_summary总结标量张量的移动平均
		add_moving_summary(
			zfill_img_MA,
			zfill_img_MB,

			recon_frq_AA, 
			recon_frq_Aa, 
			recon_frq_BB, 
			recon_frq_Bb,
			#-------lgy
			#recon_frq_aa,
			#recon_frq_bb,

			
			recon_img_AA, 
			recon_img_Aa, 
			recon_img_BB, 
			recon_img_Bb,
			#-------lgy
			#recon_img_aa,
			#recon_img_bb,


			smoothness_AA, 
			smoothness_Aa, 
			smoothness_BB, 
			smoothness_Bb,
			#-------lgy
			#smoothness_aa,
			#smoothness_bb
			)

		psnr(tf_complex(cvt2imag(M1)), tf_complex(cvt2imag(S01)), maxp=255, name='PSNR_zfill_A')
		psnr(tf_complex(cvt2imag(M2)), tf_complex(cvt2imag(S02)), maxp=255, name='PSNR_zfill_B')
		psnr(tf_complex(cvt2imag(S1)), tf_complex(cvt2imag(S01)), maxp=255, name='PSNR_recon_A')
		psnr(tf_complex(cvt2imag(S2)), tf_complex(cvt2imag(S02)), maxp=255, name='PSNR_recon_B')
		psnr(tf_complex(cvt2imag(T1)), tf_complex(cvt2imag(S01)), maxp=255, name='PSNR_boost_A')
		psnr(tf_complex(cvt2imag(T2)), tf_complex(cvt2imag(S02)), maxp=255, name='PSNR_boost_B')
		#------------------------------lgy
		#psnr(tf_complex(cvt2imag(Y1)), tf_complex(cvt2imag(S01)), maxp=255, name='PSNR_boost-lgy_A')
		#psnr(tf_complex(cvt2imag(Y2)), tf_complex(cvt2imag(S02)), maxp=255, name='PSNR_boost-lgy_B')


		def viz3(name, listTensor):
			img = tf.concat(listTensor, axis=3)
			# tensorflow中用来拼接张量的函数tf.concat()
			# axis=0 代表在第0个维度拼接 	axis=1 代表在第1个维度拼接
			# 对于一个二维矩阵，第0个维度代表最外层方括号所框下的子集，第1个维度代表内部方括号所框下的子集。维度越高，括号越小。
			# 对于[ [ ], [ ]]和[[ ], [ ]]，低维拼接等于拿掉最外面括号，高维拼接是拿掉里面的括号(保证其他维度不变)。
			out = img
			img = cvt2imag(img)
			img = tf.clip_by_value(img, 0, 255)
			
			tf.summary.image(name+'_real', tf.transpose(img[:,0:1,...], [0, 2, 3, 1]), max_outputs=50)

			tf.summary.image(name+'_imag', tf.transpose(img[:,1:2,...], [0, 2, 3, 1]), max_outputs=50)
			#tf.transpose置换 a,根据 perm 重新排列尺寸. 返回的张量的维度 i 将对应于输入维度 perm[i].
			# 如果 perm 没有给出,它被设置为(n-1 ... 0),其中 n 是输入张量的秩.
			# 因此,默认情况下,此操作在二维输入张量上执行常规矩阵转置.如果共轭为 True,并且 a.dtype 是 complex64 或 complex128,那么 a 的值是共轭转置和.
				
			return tf.identity(out, name='viz_'+name), tf.identity(img, name='vis_'+name)

		viz_A_recon, vis_A_recon = viz3('A_recon', [R, S01, M1, S1, T1, tf.abs(S01-M1), tf.abs(S01-S1), tf.abs(S01-T1), Sn1, Sp1, Tn1, Tp1])
		#viz_A_recon, vis_A_recon = viz3('A_recon',[R, S01, M1, S1, T1,Y1, tf.abs(S01-M1), tf.abs(S01-S1), tf.abs(S01-T1),tf.abs(S01-Y1), Sn1, Sp1, Tn1, Tp1, Yn1 ,Yp1])
		viz_B_recon, vis_B_recon = viz3('B_recon', [R, S02, M2, S2, T2, tf.abs(S02-M2), tf.abs(S02-S2), tf.abs(S02-T2), Sn2, Sp2, Tn2, Tp2])
		

		print (S01, R, Rh)
		print (viz_A_recon, vis_A_recon)
		print (M1, S1, T1)
	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)#get_scalar_var获取具有特定初始值的标量浮点变量。
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)
		#AdamOptimizer是TensorFlow中实现Adam算法的优化器。Adam即Adaptive Moment Estimation（自适应矩估计），
		# 是一个寻找全局最优点的优化算法，引入了二次梯度校正。Adam 算法相对于其它种类算法有一定的优越性，是比较常用的算法之一。
		#learning_rate: A Tensor or a floating point value. （学习率）
		# beta1: A float value or a constant float tensor. （一阶矩估计的指数衰减率）
		# beta2: A float value or a constant float tensor. （二阶矩估计的指数衰减率）
		# epsilon: A small constant for numerical stability. （一个非常小的数，防止除以零）
		# use_locking: 如果为真，则使用锁进行更新操作。
		# name: 使用梯度时创建的操作的可选名称，默认为 "Adam"。





###############################################################################	
def sample(imageDir, maskDir, labelDir, model_path, resultDir):
	# TODO
	print (sys.argv[0])
	pred_config = PredictConfig(
		session_init=SaverRestore(model_path), #session_init=SaverRestore(args.load)
		model=Model(),
		input_names=['inputA', 'mask', 'inputB'],
		output_names=['vis_A_recon'])


	ds_valid = ImageDataFlow(imageDir, maskDir, labelDir, 200, is_training=False)
	# ds_valid = PrefetchDataZMQ(ds_valid, nr_proc=8)
	
	filenames = glob.glob(imageDir + '/*.*')
	from natsort import natsorted
	filenames = natsorted(filenames)
	print (filenames)
	print (resultDir)

	import shutil
	shutil.rmtree(resultDir, ignore_errors=True)#递归地删除文件
	shutil.rmtree(resultDir+'/mag/', ignore_errors=True)
	shutil.rmtree(resultDir+'/ang/', ignore_errors=True)

	# Make directory to hold RefineGAN result
	os.makedirs(resultDir)#os.makedirs() 方法用于递归创建目录。像 mkdir(), 但创建的所有intermediate-level文件夹需要包含子目录。
	os.makedirs(resultDir+'/mag/')
	os.makedirs(resultDir+'/ang/')

	# Zero-filling for baseline
	os.makedirs(resultDir+'/M/')
	os.makedirs(resultDir+'/M/mag/')
	os.makedirs(resultDir+'/M/ang/')
	


	## Extract stack of images with SimpleDatasetPredictor
	pred = SimpleDatasetPredictor(pred_config, ds_valid)
	
	for idx, o in enumerate(pred.get_result()):
		print (pred)
		print (len(o))
		print (o[0].shape)

		outA = o[0][:, :, :, :] 

	
		colors0 = np.array(outA) #.astype(np.uint8)
		head, tail = os.path.split(filenames[idx])
		tail = tail.replace('png', 'tif')
		print (tail)
		print (colors0.shape)
		print (colors0.dtype)


		#np.save(resultDir+'GAN_'+tail,np.squeeze(colors0))
		import skimage.io

		skimage.io.imsave(resultDir+ "/full_"+tail, np.squeeze(colors0[...,256*1:256*2])) # Zerofill
		skimage.io.imsave(resultDir+"/zfill_"+tail, np.squeeze(colors0[...,256*2:256*3])) # Zerofill
		skimage.io.imsave(resultDir+tail, np.squeeze(colors0[...,256*4:256*5])) # Zerofill

		skimage.io.imsave(resultDir+"mag/mag_"+tail, np.abs(np_complex(np.squeeze(colors0[...,256*4:256*5]))))
		skimage.io.imsave(resultDir+"ang/ang_"+tail, np.angle(np_complex(np.squeeze(colors0[...,256*4:256*5]))))


		skimage.io.imsave(resultDir+"/M/mag/mag_"+tail, np.abs(np_complex(np.squeeze(colors0[...,256*2:256*3]))))
		skimage.io.imsave(resultDir+"/M/ang/ang_"+tail, np.angle(np_complex(np.squeeze(colors0[...,256*2:256*3]))))
        # numpy.angle计算复数的辐角主值
		#skimage.io.imsave(resultDir+"/S/mag/mag_"+tail, np.abs(np_complex(np.squeeze(colors0[...,256*3:256*4]))))
###############################################################################	
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			input_names=['inputA', 'mask', 'inputB'],
			output_names=['vis_A_recon'])

	def _before_train(self):
		global args
		self.ds_train, self.ds_valid = get_data(args.imageDir, args.maskDir, args.labelDir, size=1)
		self.ds_train.reset_state()
		self.ds_valid.reset_state() 

	def _trigger(self):
		for lst in self.ds_train.get_data():
			vis_train =  np.array(self.pred(lst)[0])
			
			vis_train_real = np.transpose(vis_train[:,0:1,...], [0, 2, 3, 1])
			vis_train_imag = np.transpose(vis_train[:,1:2,...], [0, 2, 3, 1])
			
			self.trainer.monitors.put_image('vis_train_real', vis_train_real)
			self.trainer.monitors.put_image('vis_train_imag', vis_train_imag)
		for lst in self.ds_valid.get_data():
			vis_valid = np.array(self.pred(lst)[0])
			vis_valid_real = np.transpose(vis_valid[:,0:1,...], [0, 2, 3, 1])
			vis_valid_imag = np.transpose(vis_valid[:,1:2,...], [0, 2, 3, 1])
			
			self.trainer.monitors.put_image('vis_valid_real', vis_valid_real)
			self.trainer.monitors.put_image('vis_valid_imag', vis_valid_imag)

###############################################################################		
# if __name__ == '__main__':
def main():
	np.random.seed(2018)
	tf.set_random_seed(2018)
	#https://docs.python.org/3/library/argparse.html
	parser = argparse.ArgumentParser()
	#argparse 是 Python 内置的一个用于命令项选项与参数解析的模块，通过在程序中定义好我们需要的参数，
	# argparse 将会从 sys.argv 中解析出这些参数，并自动生成帮助和使用信息。
	# 当然，Python 也有第三方的库可用于命令行解析，而且功能也更加强大，比如 docopt，Click。
	#argparse 使用
	# 简单示例
	# 我们先来看一个简单示例。主要有三个步骤：
	# 创建 ArgumentParser() 对象
	# 调用 add_argument() 方法添加参数
	# 使用 parse_args() 解析添加的参数
	parser.add_argument('--gpu',        help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load',       help='load models for continue train or predict')
	parser.add_argument('--sample',     help='run sampling one instance')
	parser.add_argument('--imageDir',   help='Image directory', required=True)
	parser.add_argument('--maskDir',    help='Masks directory', required=False)
	parser.add_argument('--labelDir',   help='Label directory', required=True)
	parser.add_argument('-db', '--debug', type=int, default=0) # Debug one particular function in main flow
	global args
	args = parser.parse_args() # Create an object of parser
	##
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
		# os.environ['TENSORPACK_TRAIN_API'] = 'v2'
	if args.sample:

		sample(args.imageDir, args.maskDir, args.labelDir, args.load, args.sample)
	else:
		logger.auto_set_dir()
		ds_train, ds_valid = get_data(args.imageDir, args.maskDir, args.labelDir)

		ds_train = PrefetchDataZMQ(ds_train, nr_proc=4)
		ds_valid = PrefetchDataZMQ(ds_valid, nr_proc=4)

		ds_train.reset_state()
		ds_valid.reset_state() 

		nr_tower = max(get_nr_gpu(), 1)
		ds_train = QueueInput(ds_train)
		model = Model()
		if nr_tower == 1:
			trainer = SeparateGANTrainer(ds_train, model, g_period=1, d_period=1)
		else:
			trainer = MultiGPUGANTrainer(nr_tower, ds_train, model)
		trainer.train_with_defaults(
			callbacks=[
				PeriodicTrigger(ModelSaver(), every_k_epochs=20),
				PeriodicTrigger(MaxSaver('validation_PSNR_recon_A'), every_k_epochs=20),
				PeriodicTrigger(MaxSaver('validation_PSNR_boost_A'), every_k_epochs=20),
				#--------------------------------lgy
				#PeriodicTrigger(MaxSaver('validation_PSNR_boost-lgy_A'), every_k_epochs=20),
				VisualizeRunner(),
				InferenceRunner(ds_valid, [
										   ScalarStats('PSNR_zfill_A'), 
										   ScalarStats('PSNR_zfill_B'),
										   ScalarStats('PSNR_recon_A'),
										   ScalarStats('PSNR_recon_B'),
										   ScalarStats('PSNR_boost_A'), 
										   ScalarStats('PSNR_boost_B'),
										   #--------------------------lgy
										   #ScalarStats('PSNR_boost-lgy_A'),
										   #ScalarStats('PSNR_boost-lgy_B'),
										
										   ScalarStats('losses/Img/Zfill/zfill_img_MA'),
										   ScalarStats('losses/Img/Zfill/zfill_img_MB'),
											  
										   ScalarStats('losses/Frq/Recon/recon_frq_AA'),
										   ScalarStats('losses/Frq/Recon/recon_frq_BB'),
										   
										   ScalarStats('losses/Img/Recon/recon_img_AA'),
										   ScalarStats('losses/Img/Recon/recon_img_BB'),
										   
										   ScalarStats('losses/Frq/Boost/recon_frq_Aa'),
										   ScalarStats('losses/Frq/Boost/recon_frq_Bb'),
										   
										   ScalarStats('losses/Img/Boost/recon_img_Aa'),
										   ScalarStats('losses/Img/Boost/recon_img_Bb'),

										   #-----------------------------------lgy
										   #ScalarStats('losses/Frq/Boost-lgy/recon_frq_aa'),
										   #ScalarStats('losses/Frq/Boost-lgy/recon_frq_bb'),

										   #ScalarStats('losses/Img/Boost-lgy/recon_img_aa'),
										   #ScalarStats('losses/Img/Boost-lgy/recon_img_bb'),
					]),
				ClipCallback(),
				ScheduledHyperParamSetter('learning_rate', 
					[(0, 1e-4), (100, 3e-4), (200, 2e-5), (300, 1e-5)], interp='linear')
				
				],
			session_init=SaverRestore(args.load) if args.load else None, 
			steps_per_epoch=ds_train.size(),
			#max_epoch=500
			max_epoch = 300
		)

