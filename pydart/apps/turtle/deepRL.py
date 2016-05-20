import tensorflow as tf
import numpy as np

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class DeepRL:
	def __init__(self, world, skel):
		self.world = world
		self.skel = skel
		self.controller = skel.controller
		self.replay_buffer = []
		self.max_buffer_size = 50000
	def step(self):
		state_skel = self.controller.get_state()
		state_eye = self.controller.get_eye().get_image()
		return []
	def run(self, max_iter=10000, max_gen_tuples=100, reset=True):
		# Reset every environments
		if reset:
			self.world.reset()
			self.controller.reset()
			del self.replay_buffer[:]
		# Start trainning
		for i in xrange(max_iter):
			# Generate trainning tuples
			for j in xrange(max_gen_tuples):
				t = self.step()
				self.replay_buffer.append(t)
				self.postprocess_replay_buffer()
			# Update the network
			data = self.sample()
	def create_model(self):
		# Check dimensions
		w,h = self.controller.get_eye().get_image_size()
		d = len(flatten(self.controller.get_state()))
		a = len(flatten(self.controller.get_action_default()))
		# 
		state_eye = tf.placeholder(tf.float32, [-1,w,h,1])
		state_skel = tf.placeholder(tf.float32, [-1,d])
		action = tf.placeholder(tf.float32, [-1,a])
		#
		W_conv1 = weight_variable([8, 8, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
	def postprocess_replay_buffer(self):
		if len(self.replay_buffer) > self.max_buffer_size:
			del self.replay_buffer[0:self.max_buffer_size]
	def reward(self):
		return np.random.uniform(0,1)
	def sample(self, sample_size=50):
		data = []
		pick_history = []
		num_data = len(self.replay_buffer)
		while num_data < sample_size:
			pick = np.random.int(num_data)
			if pick is in pick_history:
				continue
			else:
				pick_history.append(pick)
			data.append(self.replay_buffer[pick])
		return data
	def get_controller(self):
		return 0