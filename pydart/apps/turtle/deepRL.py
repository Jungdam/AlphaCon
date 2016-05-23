import tensorflow as tf
import numpy as np
import warnings

def flatten(l):
	return list(_flatten_(l))

def _flatten_(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in _flatten_(*x):
                yield y
        else:
            yield x

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
		self.gamma = 0.9
		self.sess = None
		self.model_eval_qvalue = None
		self.model_eval_action = None
		self.model_train_qvalue = None
		self.model_train_action = None
		self.model_placeholder_eye = None
		self.model_placeholder_skel = None
		self.model_placeholder_qvalue = None
		self.model_placeholder_action = None
		self.has_model = False
	def has_model(self):
		return self.has_model
	def step(self):
		self.world.reset()
		self.controller.reset()
		# here set new environment
		#
		# here set new action parameter
		sigma = 0.25
		action_default = self.controller.get_action_default()
		noiseL = np.random.normal(0.0, sigma, size=len(action_default[0]))
		noiseR = np.random.normal(0.0, sigma, size=len(action_default[1]))
		noiseT = np.random.normal(0.0, sigma, size=1)
		l = np.array(action_default[0]) + noiseL
		r = np.array(action_default[1]) + noiseR
		t = np.array(action_default[2]) + noiseT
		action = [l.tolist(),r.tolist(),t[0]]
		self.controller.add_action(action)
		state_skel_init = np.array(self.controller.get_state())
		state_eye_init = self.controller.get_eye().get_image()
		while True:
			self.world.step()
			if self.controller.is_new_wingbeat():
				break
		state_skel_term = np.array(self.controller.get_state())
		state_eye_term = self.controller.get_eye().get_image()
		reward = self.reward()
		return [state_skel_init, state_eye_init, action, reward, state_skel_term, state_eye_term]
	def run(self, max_iter=50000, data_per_iter=100, reset=True):
		if self.has_model is False:
			warnings.warn('DeepRL: No Model is created')
			return
		# Reset every environments
		if reset:
			del self.replay_buffer[:]
		# Start trainning
		for i in xrange(max_iter):
			# Generate trainning tuples
			for j in xrange(data_per_iter):
				t = self.step()
				self.replay_buffer.append(t)
			self.postprocess_replay_buffer()
			# Update the network
			data = self.sample()
			self.update_model(data)
	def update_model(self, data):
		if self.has_model is False:
			warnings.warn('DeepRL: No Model is created')
			return
		data_state_eye = []
		data_state_skel = []
		data_action = []
		data_reward = []
		data_state_eye_prime = []
		data_state_skel_prime = []
		
		for episode in data:
			data_state_eye.append(data[0])
			data_state_skel.append(data[1])
			data_action.append(data[2])
			data_reward.append(data[3])
			data_state_eye_prime.append(data[4])
			data_state_skel_prime.append(data[5])

		data_state_eye = np.array(data_state_eye)
		data_state_skel = np.array(data_state_skel)
		data_action = np.array(data_action)
		data_reward = np.array(data_reward)
		data_state_eye_prime = np.array(data_state_eye_prime)
		data_state_skel_prime = np.array(data_state_skel_prime)

		qvalue = self.model_eval_qfnc.eval(feed_dict={ \
			self.model_placeholder_eye: data_state_eye, \
			self.model_placeholder_skel: data_state_skel})
		qvalue_prime = self.model_eval_qfnc.eval(feed_dict={ \
			self.model_placeholder_eye: data_state_eye_prime, \
			self.model_placeholder_skel: data_state_skel_prime})
		action = self.model_eval_act.eval(feed_dict={ \
			self.model_placeholder_eye: data_state_eye, \
			self.model_placeholder_skel: data_state_skel})

		target_qvalue = data_reward + self.gamma*qvalue_prime
		target_action = data_action

		self.model_train_qfnc.run(feed_dict={ \
			self.model_placeholder_eye: data_state_eye, \
			self.model_placeholder_skel: data_state_skel, \
			self.model_placeholder_qvalue: data_target_qvalue})
		self.model_train_act.run(feed_dict={ \
			self.model_placeholder_eye: data_state_eye, \
			self.model_placeholder_skel: data_state_skel, \
			self.model_placeholder_action: data_target_action})
	def create_model(self):
		# Check dimensions
		w,h = self.controller.get_eye().get_image_size()
		print flatten(self.controller.get_state())
		print flatten(self.controller.get_action_default())
		d = len(flatten(self.controller.get_state()))
		a = len(flatten(self.controller.get_action_default()))
		# 
		state_eye = tf.placeholder(tf.float32, [None,w,h,1])
		state_skel = tf.placeholder(tf.float32, [None,d])
		qvalue = tf.placeholder(tf.float32, [None,1])
		action = tf.placeholder(tf.float32, [None,a])
		# Frist conv layer for the eye
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(state_eye, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
		# Second conv layer for the eye
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)
		# Fully connected layer for the eye
		W_fc1 = weight_variable([(w/4)*(h/4)*64, 256])
		b_fc1 = bias_variable([256])
		h_pool2_flat = tf.reshape(h_pool2, [-1, (w/4)*(h/4)*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
		# Combined layer for the eye and the skel
		W_fc2 = weight_variable([256+d, 512])
		b_fc2 = bias_variable([512])
		h_comb1 = tf.concat(1, [h_fc1, state_skel])
		h_fc2 = tf.nn.relu(tf.matmul(h_comb1, W_fc2) + b_fc2)
		#
		W_fc3_qfnc = weight_variable([512, 1])
		b_fc3_qfnc = bias_variable([1])
		h_fc3_qfnc = tf.matmul(h_fc2, W_fc3_qfnc) + b_fc3_qfnc
		W_fc3_act = weight_variable([512, a])
		b_fc3_act = bias_variable([a])
		h_fc3_act = tf.matmul(h_fc2, W_fc3_act) + b_fc3_act
		# Optimizer
		loss_qfnc = tf.reduce_mean(tf.square(qvalue - h_fc3_qfnc))
		optimizer_qfnc = tf.train.GradientDescentOptimizer(0.1)
		self.model_train_qvalue = optimizer_qfnc.minimize(loss_qfnc)
		loss_act = tf.reduce_mean(tf.square(action - h_fc3_act))
		optimizer_act = tf.train.GradientDescentOptimizer(0.1)
		self.model_train_action = optimizer_qfnc.minimize(loss_act)
		# Place holders
		self.model_placeholder_eye = state_eye
		self.model_placeholder_skel = state_skel
		self.model_placeholder_qvalue = qvalue
		self.model_placeholder_action = action
		# Evaultion
		self.model_eval_qvalue = h_fc3_qfnc
		self.model_eval_action = h_fc3_act
		# Initialize all variables
		init = tf.initialize_all_variables()
		self.sess = tf.InteractiveSession()
		self.sess.run(init)

		self.has_model = True
	def postprocess_replay_buffer(self):
		max_size = self.max_buffer_size
		cur_size = len(self.replay_buffer)
		if cur_size > max_size:
			del self.replay_buffer[0:cur_size-max_size]
	def reward(self):
		return np.random.uniform(0,1)
	def sample(self, sample_size=50):
		data = []
		pick_history = []
		num_data = len(self.replay_buffer)
		while num_data < sample_size:
			pick = np.random.int(num_data)
			if pick in pick_history:
				continue
			else:
				pick_history.append(pick)
			data.append(self.replay_buffer[pick])
		return data
	def get_controller(self):
		return 0