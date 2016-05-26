import tensorflow as tf
import warnings

name_list = []

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.02, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class NN:
	def __init__(self, name):
		if name is in name_list:
			warnings.warn('NN: duplicated name / '+name)
			return
		self.name = name
		tf.reset_default_graph()
		self.graph = tf.Graph()
		self.sess = None
	def not_implemented(self, func_name):
		print '[', func_name, ']', 'function is not implemented'
	def create(self):
		self.not_implemented('create')
	def train(self):
		self.not_implemented('train')