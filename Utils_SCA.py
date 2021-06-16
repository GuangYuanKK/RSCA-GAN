#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob


# Misc. libraries
from six.moves import map, zip, range
#from natsort import natsorted

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation


# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.utils import logger
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

# Tensorflow 1
import tensorflow as tf
from tensorflow import layers
# from tensorflow.contrib.layers.python import layers
###############################################################################
SHAPE = 256
BATCH = 1
TEST_BATCH = 100
EPOCH_SIZE = 100
NB_FILTERS = 64  # channel size频道宽度; 通道大小
UCA_PARA = 32#reshape用的
DIMX  = 256
DIMY  = 256
DIMZ  = 2
DIMC  = 1
###############################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)#计算激活函数 relu,即 max(features, 0)。将大于0的保持不变,小于0的数置为0
###############################################################################
def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)#计算Leaky ReLU激活函数.
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)
###############################################################################
# Utility function for scaling 
def cvt2tanh(x, name='ToRangeTanh'):
	with tf.variable_scope(name):#用于定义创建变量（层）的操作的上下文管理器
		# 此上下文管理器验证（可选）values是否来自同一图形，确保图形是默认的图形，并推送名称范围和变量范围。
		return (x / 255.0 - 0.5) * 2.0
###############################################################################
def cvt2imag(x, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * 255.0
###############################################################################		
def cvt2sigm(x, name='ToRangeSigm'):
	with tf.variable_scope(name):
		return (x / 1.0 + 1.0) / 2.0
###############################################################################
def tf_complex(data, name='tf_channel'):
	with tf.variable_scope(name+'_scope'):
		real  = data[:,0:1,...]
		imag  = data[:,1:2,...]
		del data
		data  = tf.complex(real, imag)  # tf.complex将两实数转换为复数形式
	data = tf.identity(data, name=name)
	#它返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作
	return data	
###############################################################################
def tf_channel(data, name='tf_complex'):
	with tf.variable_scope(name+'_scope'):
		real  = tf.real(data)#返回TensorFlow复数的实数部分
		imag  = tf.imag(data)#返回TensorFlow虚数的实数部分
		real  = real[:,0:1,...]
		imag  = imag[:,0:1,...]
		del data
		data  = tf.concat([real, imag], axis=1)
		#tensorflow中用来拼接张量的函数tf.concat()
		#axis=0 代表在第0个维度拼接 	axis=1 代表在第1个维度拼接
		#对于一个二维矩阵，第0个维度代表最外层方括号所框下的子集，第1个维度代表内部方括号所框下的子集。维度越高，括号越小。
		#对于[ [ ], [ ]]和[[ ], [ ]]，低维拼接等于拿掉最外面括号，高维拼接是拿掉里面的括号(保证其他维度不变)。
	data = tf.identity(data, name=name)
	# 它返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作
	return data
###############################################################################
def np_complex(data):
	real  = data[0,...]
	imag  = data[1,...]
	del data
	data = real + 1j*imag
	return data	

###############################################################################
def np_channel(data):
	real  = np.real(data)
	imag  = np.imag(data)
	del data
	data  = np.concatenate([real, imag], axis=1)#同tf.concat
	return data		

###############################################################################
# tfutils.symbolic_functions.psnr(prediction, ground_truth, maxp=None, name='psnr')
def psnr(prediction, ground_truth, maxp=None, name='psnr'):
	"""`Peek Signal to Noise Ratio <https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio>`_.

	.. math::

		PSNR = 20 \cdot \log_{10}(MAX_p) - 10 \cdot \log_{10}(MSE)

	Args:
		prediction: a :class:`tf.Tensor` representing the prediction signal.
		ground_truth: another :class:`tf.Tensor` with the same shape.
		maxp: maximum possible pixel value of the image (255 in in 8bit images)

	Returns:
		A scalar tensor representing the PSNR.
	"""
	prediction   = tf.abs(prediction)
	ground_truth = tf.abs(ground_truth)
	def log10(x):
		with tf.name_scope("log10"):
			#在某个tf.name_scope()指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域；
			#将不同的对象及操作放在由tf.name_scope()指定的区域中，便于在tensorboard中展示清晰的逻辑关系图，这点在复杂关系图中特别重要。
			numerator = tf.log(x)
			denominator = tf.log(tf.constant(10, dtype=numerator.dtype))#tf.constant创建常量的函数
			return numerator / denominator

	mse = tf.reduce_mean(tf.square(prediction - ground_truth))
	#tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。
	#tf.square对x内的所有元素进行平方操作
	if maxp is None:
		psnr = tf.multiply(log10(mse), -10., name=name)#tf.multiply（）两个矩阵中对应元素各自相乘
	else:
		maxp = float(maxp)
		psnr = tf.multiply(log10(mse+1e-6), -10.)
		psnr = tf.add(tf.multiply(20., log10(maxp)), psnr, name=name)#tf.add 计算张量的和,上述操作返回 x + y 元素
	add_moving_summary(psnr)
	return psnr
			
###############################################################################
def RF(image, mask, name="RF"):
	# This op perform undersampling
	with tf.variable_scope(name+'_scope'):
		# Convert from 2 channel to complex number
		image = tf_complex(image)
		mask  = tf_complex(mask) 

		# Forward Fourier Transform
		freq_full = tf.fft2d(image, name='Ff')#二维离散傅立叶变换函数
		freq_zero = tf.zeros_like(freq_full)#填充的数据是0
		condition = tf.cast(tf.real(mask)>0.9, tf.bool)
		#tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，
		# 比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32

		freq_dest = tf.where(condition, freq_full, freq_zero, name='RfFf')
		#根据condition返回x或y中的元素.
		# 如果x和y都为None,则该操作将返回condition中true元素的坐标.坐标以二维张量返回,其中第一维(行)表示真实元素的数量,第二维(列)表示真实元素的坐标.
		# 请记住,输出张量的形状可以根据输入中的真实值的多少而变化.索引以行优先顺序输出.
		# 如果两者都不是None,则x和y必须具有相同的形状.如果x和y是标量,则condition张量必须是标量.
		# 如果x和y是更高级别的矢量,则condition必须是大小与x的第一维度相匹配的矢量,或者必须具有与x相同的形状.、
		# condition张量作为一个可以选择的掩码(mask),它根据每个元素的值来判断输出中的相应元素/行是否应从 x (如果为 true) 或 y (如果为 false)中选择.、
		# 如果condition是向量,则x和y是更高级别的矩阵,那么它选择从x和y复制哪个行(外部维度).如果condition与x和y具有相同的形状,那么它将选择从x和y复制哪个元素.
		# Convert from complex number to 2 channel
		freq_dest = tf_channel(freq_dest)
	return tf.identity(freq_dest, name=name)#它返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作

###############################################################################
def FhRh(freq, mask, name='FhRh', is_normalized=False):
	with tf.variable_scope(name+'_scope'):
		# Convert from 2 channel to complex number
		freq = tf_complex(freq)
		mask = tf_complex(mask) 

		# Under sample
		condition = tf.cast(tf.real(mask)>0.9, tf.bool)
		# tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换，
		# 比如读入的图片如果是int8类型的，一般在要在训练前把图像的数据格式转换为float32
		freq_full = freq
		freq_zero = tf.zeros_like(freq_full)#填充的数据是0
		freq_dest = tf.where(condition, freq_full, freq_zero, name='RfFf')

		# Inverse Fourier Transform
		image 	  = tf.ifft2d(freq_dest, name='FtRt')#二维离散傅立叶变换函数
		
		if is_normalized:
			image = tf.div(image, ((DIMX-1)*(DIMY-1)))#除

		# Convert from complex number to 2 channel
		image = tf_channel(image)
	return tf.identity(image, name)#它返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作

###############################################################################
def update(recon, image, mask, name='update'):
	"""
	Update the reconstruction with undersample k-space measurement
	"""
	with tf.variable_scope(name+'_scope'):
		k_recon = RF(recon, tf.ones_like(mask), name='k_recon')#tf.ones_like创建一个将所有元素设置为1的张量。
		k_image = RF(image, tf.ones_like(mask), name='k_image')

		m_real = mask[:,0:1,...]
		m_imag = mask[:,0:1,...]
		m_mask = tf.concat([m_real, m_imag], axis=1)#tensorflow中用来拼接张量的函数tf.concat()
		print (mask, k_recon, k_image)
		condition = tf.cast(tf.real(m_mask)>0.9, tf.bool)
		# where(
		#     condition,
		#     x=None,
		#     y=None,
		#     name=None
		# )
		#Return the elements, either from x or y, depending on the condition.
		k_return  = tf.where(condition, k_image, k_recon, name='k_return')
		updated = FhRh(k_return, tf.ones_like(mask), name=name)
	return tf.identity(updated, name=name)#它返回一个和输入的 tensor 大小和数值都一样的 tensor ,类似于 y=x 操作

###############################################################################
def ChannelWiseAttention(x: tf.Tensor, name: str):
    """
    通道注意力转移
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        _, C, H, W = x.get_shape()
        w = tf.get_variable("attention_w", [C, C], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(initializer=tf.constant(0.0, dtype=tf.float32, shape=[C]), trainable=True, name='attention_b')
        transpose_feature_map = tf.transpose(tf.reduce_mean(x, [2, 3], keep_dims=True), perm=[0, 1, 2, 3])  # 每一个通道
        channel_wise_attention = tf.matmul(tf.reshape(transpose_feature_map,  [-1, C]), w) + b  # b, c
        channel_wise_attention = tf.nn.sigmoid(channel_wise_attention)
        channel_wise_attention = tf.tile(input=channel_wise_attention, multiples=[1, H*W])
        attention = tf.reshape(channel_wise_attention, [-1, C, H, W])
        attention_x = tf.multiply(x=x, y=attention)
		# np.save('/home1/wangcy/JunLyu/lgy/RefineGAN_brain/attention/ca/ca.npy', attention_x)
        return attention_x


def SpatialAttention(x: tf.Tensor, name: str, k: int=1024):
    """
    空间注意力转移
    :param x:  [batch_size, height, width, channel]
    :param name:
    :param k:
    :return:
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        _, C, H, W = x.get_shape()
        w = tf.get_variable(name="attention_w", shape=[C, 1], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        b = tf.get_variable(initializer=tf.constant(0.0, dtype=tf.float32, shape=[1]), trainable=True, name='attention_b')
        spatial_attention = tf.matmul(tf.reshape(x, [-1, C]), w) + b  # 每一个空间点位置的attention  多个通道的同一个位置 生成一个概率
        spatial_attention = tf.nn.sigmoid(tf.reshape(spatial_attention, [-1, W * H]))  # batch_size, w*h
        spatial_attention = tf.tile(input=spatial_attention, multiples=[1, C])  # batch_size, w*h*c
        attention = tf.reshape(spatial_attention, [-1, C, H, W])  # batch_size, height, w, channel
        attention_x = tf.multiply(x=x, y=attention)
		# np.save('/home1/wangcy/JunLyu/lgy/RefineGAN_brain/attention/sa/sa.npy', attention_x)
		# i = 1
		# sess = tf.Session()
		# sa0_numpy = attention_x.eval(session=sess)
		# np.save('/home1/wangcy/JunLyu/lgy/RefineGAN_brain/attention/sa/sa_'+str(i)+'.npy',sa0_numpy);
		# i = i+1
        return attention_x
###############################################################################
# FusionNet
@layer_register(log_shape=True)
# def residual(x, chan, first=False):
# 	with argscope([Conv2D], stride=1, kernel_shape=3):
# 		input = x
# 		return (LinearWrap(x)
# 				.Conv2D('conv0', chan, padding='SAME')
# 				# .Dropout('drop', 0.5)
# 				.Conv2D('conv1', chan/2, padding='SAME')
# 				.Conv2D('conv2', chan, padding='SAME', nl=tf.identity)
# 				# .Dropout('drop', 0.5)
# 				# .InstanceNorm('inorm')
# 				()) + input
def residual(x, chan, first=False):
	with argscope([Conv2D], stride=1, kernel_shape=3):
		input = x
		output = (LinearWrap(x)
				.Conv2D('conv0', chan, padding='SAME')
				# .Dropout('drop', 0.5)
				.Conv2D('conv1', chan/2, padding='SAME')
				.Conv2D('conv2', chan, padding='SAME', nl=tf.identity)
				# .Dropout('drop', 0.5)
				# .InstanceNorm('inorm')
				())
		sa = SpatialAttention(output, 'sa')
		ca = ChannelWiseAttention(sa, 'ca')
		return ca + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=1, stride=1):
	with argscope([Conv2D], stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		#给定tensor,这个操作返回一个张量,它与带有形状shape的tensor具有相同的值.
		#如果shape的一个分量是特殊值-1,则计算该维度的大小,以使总大小保持不变.特别地情况为,一个[-1]维的shape变平成1维.至多能有一个shape的分量可以是-1.
		#如果shape是1-D或更高,则操作返回形状为shape的张量,其填充为tensor的值.在这种情况下,隐含的shape元素数量必须与tensor元素数量相同.
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.9)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_enc1', chan, first=True)#残差
			.residual('res_enc2', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			# .InstanceNorm('inorm')
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Deconv2D('deconv_i', chan, stride=1) 
			.residual('res_dec1', chan, first=True)
			.residual('res_dec2', chan, first=True)
			.Deconv2D('deconv_o', chan, stride=2) 
			# .InstanceNorm('inorm')
			# .Dropout('drop', 0.9)
			())
		return x
############################################################


###############################################################################
@auto_reuse_variable_scope
def arch_generator(img):
	assert img is not None
	# img = tf_complex(img)
	with argscope([Conv2D, Deconv2D], nl=BNLReLU, kernel_shape=4, stride=2, padding='SAME'):
		#deconv解卷积，实际是叫做conv_transpose, conv_transpose实际是卷积的一个逆向过程
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		# e0 = Dropout('dr', e0, 0.9)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		# e3 = Dropout('dr', e3, 0.9)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		#se3 = channel_wise_attention('se3',d3, NB_FILTERS*4,32, 32, 8)
		sa3 = SpatialAttention(d3, 'sa3')
		# cwa3 = channel_wise_attention(sa3,d3,256,'cwa3')
		cwa3 = ChannelWiseAttention(sa3, 'cwa3')
		# cwa3 = ChannelWiseAttention(input_x=sa3, out_dim=NB_FILTERS * 4, ratio=reduction_ratio, name="cwa3")
		# sa3 = SpatialAttention(cwa3,'sa3')
		#d2 = residual_dec('d2', d3 + e2, NB_FILTERS * 2)
		d2 = residual_dec('d2', cwa3+e2, NB_FILTERS*2)
		#se2 = channel_wise_attention('se2',d2,NB_FILTERS*2, 64, 64, 8)
		sa2 = SpatialAttention(d2, 'sa2')
		# cwa2 = channel_wise_attention(sa2,d2,256, 'cwa2')
		cwa2 = ChannelWiseAttention(sa2, 'cwa2')
		# cwa2 = ChannelWiseAttention(input_x=sa2, out_dim=NB_FILTERS * 2, ratio=reduction_ratio, name="cwa2")
		# sa2 = SpatialAttention(cwa2, 'sa2')
		d1 = residual_dec('d1', cwa2+e1, NB_FILTERS*1)
		#se1 = channel_wise_attention('se1',d1,NB_FILTERS*1, 128, 128, 8)
		sa1 = SpatialAttention(d1, 'sa1')
		# cwa1 = channel_wise_attention(sa1,d1,256, 'cwa1')
		cwa1 = ChannelWiseAttention(sa1,'cwa1')
		# cwa1 = ChannelWiseAttention(input_x=sa1, out_dim=NB_FILTERS * 1, ratio=reduction_ratio, name="cwa1")
		# sa1 = SpatialAttention(cwa1, 'sa1')
		d0 = residual_dec('d0', cwa1+e0, NB_FILTERS*1)
		#se0 = channel_wise_attention('se0',d0,NB_FILTERS*1,256, 256, 8)
		sa0 = SpatialAttention(d0, 'sa0')
		# sa = sa0
		# i = 1
		# with tf.Session() as sess:
		# sess = tf.Session()
		# sa0_numpy = sa.eval(session=sess)
		# np.save('/home1/wangcy/JunLyu/lgy/RefineGAN_brain/attention/sa/sa_'+str(i)+'.npy',sa0_numpy);
		# cwa0 = channel_wise_attention(sa0,d0,256, 'cwa0')
		cwa0 = ChannelWiseAttention(sa0,'cwa0')
		# ca = cwa0
		# with tf.Session() as sess:
		# cwa0_numpy = ca.eval(session=sess)
		# np.save('/home1/wangcy/JunLyu/lgy/RefineGAN_brain/attention/ca/ca_'+str(i)+'.npy',cwa0_numpy);
		# i=i+1
		# cwa0 = ChannelWiseAttention(input_x=sa0, out_dim=NB_FILTERS * 1, ratio=reduction_ratio, name="cwa0")
		# sa0 = SpatialAttention(cwa0, 'sa0')
		dd =  (LinearWrap(cwa0)
				.Conv2D('convlast', 2, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		l  = (dd)
		return l

###############################################################################
# @auto_reuse_variable_scope
def arch_discriminator(img):
	assert img is not None
	# img = tf_complex(img)
	with argscope([Conv2D, Deconv2D], nl=BNLReLU, kernel_shape=4, stride=2, padding='SAME'):
		img = Conv2D('conv0', img, NB_FILTERS, nl=tf.nn.leaky_relu)
		# img = Dropout('dr', img, 0.9)
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)

		ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
		return ret

###############################################################################
class ClipCallback(Callback):
	def _setup_graph(self):
		vars = tf.trainable_variables()
		#tf.trainable_variables这个函数可以也仅可以查看可训练的变量
		#对于一些我们不需要训练的变量，比较典型的例如学习率或者计步器这些变量，我们都需要将trainable设置为False，
		# 这时tf.trainable_variables() 就不会打印这些变量。
		ops = []
		for v in vars:
			n = v.op.name
			if not n.startswith('discrim/'):
				continue
			logger.info("Clip {}".format(n))
			ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))#tf.clip_by_value截取V使之在min和max之间
		self._op = tf.group(*ops, name='clip')#tf.group用于创造一个操作,可以将传入参数的所有操作进行分组

	def _trigger_step(self):
		self._op.run()
###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, maskDir, labelDir, size, ratio = 0.1, dtype='float32', is_training=False):
		"""
		Args:
			shapes (list): a list of lists/tuples. Shapes of each component.
			size (int): size of this DataFlow.
			random (bool): whether to randomly generate data every iteration.
				Note that merely generating the data could sometimes be time-consuming!
			dtype (str): data type.
		"""
		# super(FakeData, self).__init__()

		self.dtype    = dtype
		self.imageDir = imageDir
		self.maskDir  = maskDir
		self.labelDir = labelDir
		self.ratio    = ratio
		self._size    = size
		self.is_training = is_training
	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)   
		print (self.is_training)


	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))#ndim对于一个数组，其shape属性的长度（length）也既是它的ndim.
		if seed:
			np.random.seed(seed)
		random_flip = np.random.randint(1,5)#函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
											# 如果没有写参数high的值，则返回[0,low)的值。
		if random_flip==1:
			flipped = image[...,::1,::-1]
			#image[:,:,::-1]opencv读取图像时，图片的颜色通道为 GBR，为了与原始图片的RGB通道同步，需要转换颜色通道
			#image[:,::-1,:] 水平翻转
			#image[::-1,:,:] 上下翻转
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_reverse = np.random.randint(1,3)#函数的作用是，返回一个随机整型数，范围从低（包括）到高（不包括），即[low, high)。
											# 如果没有写参数high的值，则返回[0,low)的值。
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]

		return reverse

	def random_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = np.random.randint(-90,90)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		rotated = rotate(image, random_rotatedeg, axes=(1,2), reshape=False)#rotate旋转图片
		image = rotated
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image
		
	def random_crop(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			#随机数种子对后面的结果一直有影响。同时，加了随机数种子以后，后面的随机数组都是按一定的顺序生成的
		limit = np.random.randint(1, 12) # Crop pixel
		randy = np.random.randint(0, limit)
		randx = np.random.randint(0, limit)
		cropped = image[:, randy:-(limit-randy), randx:-(limit-randx)]
		return cropped	
	##################################################################
	def get_data(self, shuffle=True):
		# self.reset_state()
		images = glob.glob(self.imageDir + '/*.*')
		# print "images: ", images
		if self.maskDir:
			masks  = glob.glob(self.maskDir + '/*.*')
		# print "masks: ", masks
		labels = glob.glob(self.labelDir + '/*.*')
		# print "labels: ", labels
		from natsort import natsorted
		images = natsorted(images)
		if self.maskDir:
			masks  = natsorted(masks)
		labels = natsorted(labels)
		# print images
		# print labels

		for k in range(self._size):
			if self.is_training:
				from random import randrange
				rand_index_image = randrange(0, len(images))
				#randrange() 方法返回指定递增基数集合中的一个随机数，基数默认值为1
				if self.maskDir:
					rand_index_mask  = randrange(0, len(masks))
				rand_index_label = randrange(0, len(labels))
				# rand_index = randrange(0, len(images))
			else:
				rand_index_image = k
				rand_index_mask  = 0
				rand_index_label = k

			image = skimage.io.imread(images[rand_index_image])
			if self.maskDir:
				mask  = skimage.io.imread(masks[rand_index_mask])#读取图片
			else:
				mask = 255*self.generateMask(DIMZ, DIMY, DIMX, sampling_rate=self.ratio)
			label = skimage.io.imread(labels[rand_index_label])
			
			# print images[rand_index_image], masks[rand_index_mask], labels[rand_index_label]
			# print image.shape, mask.shape, label.shape

			# # Process the static image, make 2 channel image identical
			if image.ndim == 2:
				image = np.stack((image, np.zeros_like(image)), axis=0)
				#np.stack对指定axis增加维度 例(3,3)的矩阵
				# 当给x1的axis = 0也就是第一维增加一维后就变成了（2,3,3）
				# 当axis = 1时，对二维平面的行进行增加 (3,2,3)
				# 当axis = 2时，对三维平面的行进行增加 (3,3,2)
			if mask.ndim == 2:
				mask = np.stack((mask, np.zeros_like(mask)), axis=0)
			if label.ndim == 2:
				label = np.stack((label, np.zeros_like(label)), axis=0)



			seed_image = np.random.randint(0, 2015)
			seed_mask  = np.random.randint(0, 2015)
			seed_label = np.random.randint(0, 2015)

			if self.is_training:
				# pass
				#TODO: augmentation here	

				image = self.random_square_rotate(image, seed=seed_image)
				image = self.random_flip(image, seed=seed_image)
				image = self.random_crop(image, seed=seed_image)


				label = self.random_square_rotate(label, seed=seed_label)
				label = self.random_flip(label, seed=seed_label)
				label = self.random_crop(label, seed=seed_label)


			image = skimage.transform.resize(image, output_shape=(DIMZ, DIMY, DIMX), mode='constant',
											 order=1, preserve_range=True)
			#skimage.transform.resize图像的形变与缩放
			# skimage.transform.resize(image, output_shape)
			# image: 需要改变尺寸的图片
			# output_shape: 新的图片尺寸
			label = skimage.transform.resize(label, output_shape=(DIMZ, DIMY, DIMX), mode='constant',
											 order=1, preserve_range=True)

			image = np.expand_dims(image, axis=0)
			mask  = np.expand_dims(mask, axis=0)
			label = np.expand_dims(label, axis=0)



			# yield [image.astype(np.complex64), mask.astype(np.complex64), label.astype(np.complex64)]
			yield [image.astype(np.uint8), 
				   mask.astype(np.uint8), 
				   label.astype(np.uint8)]


def get_data(imageDir, maskDir, labelDir, size=EPOCH_SIZE):
	ds_train = ImageDataFlow(imageDir, 
							 maskDir, 
							 labelDir, 
							 size, 
							 ratio=0.1,
							 is_training=True
							 )


	ds_valid = ImageDataFlow(imageDir.replace('train', 'valid'), 
							 maskDir, 
							 labelDir.replace('train', 'valid'), 
							 size, 
							 ratio=0.1,
							 is_training=False
							 )

	return ds_train, ds_valid

	