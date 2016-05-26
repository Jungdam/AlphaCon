import tensorflow as tf
import numpy as np
import warnings
import scene
import math

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
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.02, shape=shape)
  return tf.Variable(initial)
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def add_action(a, b):
	return list([
		(np.array(a[0])+np.array(b[0])).tolist(),
		(np.array(a[1])+np.array(b[1])).tolist(),
		a[2]+b[2]])
def sub_action(a, b):
	return list([
		(np.array(a[0])-np.array(b[0])).tolist(),
		(np.array(a[1])-np.array(b[1])).tolist(),
		a[2]-b[2]])

class DeepRL:
	def __init__(self, world, skel, scene):
		self.world = world
		self.skel = skel
		self.controller = skel.controller
		self.scene = scene
		self.replay_buffer = []
		self.buffer_size = 0
		self.buffer_size_accum = 0
		self.max_buffer_size = 10000
		self.model_initialized = False
		self.gamma = 0.9
		self.init_exprolation_prob = 0.5
		self.max_episode_generation = 100000
		self.dropout_keep_prob = 0.5
		self.sess = None
		self.model_eval_qvalue = None
		self.model_eval_action = None
		self.model_train_qvalue = None
		self.model_train_action = None
		self.model_placeholder_eye = None
		self.model_placeholder_skel = None
		self.model_placeholder_qvalue = None
		self.model_placeholder_action = None
		self.model_loss_qvalue = None
		self.model_loss_action = None
		self.model_dropout_keep_prob = None
	def has_model(self):
		return self.model_initialized
	def init_step(self):
		# initialize world and controller
		self.world.reset()
		self.controller.reset()
		# set new environment
		self.scene.perturbate()
		self.scene.update(self.skel.body('trunk').T)
	def get_max_episode_generation(self):
		return self.max_episode_generation
	def get_exprolation_prob(self):
		sigma_inv = 2.5
		return self.init_exprolation_prob * \
			math.exp(-sigma_inv*float(self.buffer_size_accum)/float(self.max_episode_generation))
	def step(self):
		eye = self.controller.get_eye()
		# 
		state_eye_init = eye.get_image(self.skel.body('trunk').T)
		state_skel_init = self.controller.get_state()
		reward_init = self.reward()
		if self.check_terminatation(state_eye_init):
			return None
		# set new action parameter
		action_default = self.controller.get_action_default()
		action_extra = self.eval_action(state_eye_init, state_skel_init)
		action_extra = self.perturbate_action(action_extra, [self.get_exprolation_prob()]*len(flatten(action_default)))
		action = add_action(action_default, action_extra)
		# print action
		self.controller.add_action(action)
		action_extra_flat = flatten(action_extra)
		while True:
			self.world.step()
			self.scene.update(self.skel.body('trunk').T)
			if self.controller.is_new_wingbeat():
				break
		state_eye_term = eye.get_image(self.skel.body('trunk').T)
		state_skel_term = self.controller.get_state()
		reward_term = self.reward()
		reward = reward_term - reward_init
		
		return [state_eye_init, state_skel_init, action_extra_flat, reward, state_eye_term, state_skel_term]
	def perturbate_action(self, action, sigma):
		action_flat = flatten(action)
		for i in xrange(len(action_flat)):
			action_flat[i] += np.random.normal(0.0, sigma[i])
		return [action_flat[0:6],action_flat[6:12],action_flat[12]]
	def reset_buffer(self):
		if reset:
			del self.replay_buffer[:]
			self.buffer_size = 0
			self.buffer_size_accum = 0
	def get_buffer_size_accumulated(self):
		return self.buffer_size_accum
	def run(self, max_episode=100, max_iter=100):
		if self.has_model is False:
			warnings.warn('DeepRL: No Model is created')
			return
		# Start trainning
		for i in xrange(max_episode):
			print '[ ', i, 'th episode ]',\
				'buffer_size:',self.buffer_size,\
				'buffer_acum:',self.buffer_size_accum, 

			self.init_step()
			# Generate trainning tuples
			for j in xrange(max_iter):
				# print '\r', j, 'th iteration'
				t = self.step()
				# print t is None
				if t is None:
					self.init_step()
				else:
					self.replay_buffer.append(t)
					self.buffer_size += 1
					self.buffer_size_accum += 1
			self.postprocess_replay_buffer()
			# Update the network
			data = self.sample()
			self.update_model(data, True)
	def check_terminatation(self, state_eye):
		# No valid depth image
		threshold = 0.99
		w,h,c = state_eye.shape
		term_eye = True
		for i in xrange(w):
			if term_eye is False:
				break
			for j in xrange(h):
				val = state_eye[i,j,0]
				if val <= threshold:
					term_eye = False
					break
		if term_eye:
			return True
		# Passed every termination conditions
		return False
	def update_model(self, data, print_loss=False):
		if self.model_initialized is False:
			warnings.warn('DeepRL: No Model is created')
			return
		data_state_eye = []
		data_state_skel = []
		data_action = []
		data_reward = []
		data_state_eye_prime = []
		data_state_skel_prime = []

		for episode in data:
			data_state_eye.append(episode[0])
			data_state_skel.append(episode[1])
			data_action.append(episode[2])
			data_reward.append([episode[3]])
			data_state_eye_prime.append(episode[4])
			data_state_skel_prime.append(episode[5])

		data_state_eye = np.array(data_state_eye)
		data_state_skel = np.array(data_state_skel)
		data_action = np.array(data_action)
		data_reward = np.array(data_reward)
		data_state_eye_prime = np.array(data_state_eye_prime)
		data_state_skel_prime = np.array(data_state_skel_prime)

		# print data_state_eye.shape
		# print data_state_skel.shape
		# print data_action.shape
		# print data_reward.shape

		qvalue_prime = self.model_eval_qvalue.eval(feed_dict={
			self.model_placeholder_eye: data_state_eye_prime,
			self.model_placeholder_skel: data_state_skel_prime,
			self.model_dropout_keep_prob: 1.0})
		# qvalue = self.model_eval_qvalue.eval(feed_dict={
		# 	self.model_placeholder_eye: data_state_eye,
		# 	self.model_placeholder_skel: data_state_skel})
		# action = self.model_eval_action.eval(feed_dict={
		# 	self.model_placeholder_eye: data_state_eye,
		# 	self.model_placeholder_skel: data_state_skel})

		# print flatten(qvalue)
		# print flatten(qvalue_prime)
		# print action

		target_qvalue = data_reward + self.gamma*qvalue_prime
		target_action = data_action

		# print data_reward
		# print qvalue_prime
		# print self.gamma*qvalue_prime
		# print target_qvalue
		# print target_action

		if print_loss:
			q = self.model_loss_qvalue.eval(feed_dict={
				self.model_placeholder_eye: data_state_eye,
				self.model_placeholder_skel: data_state_skel,
				self.model_placeholder_qvalue: target_qvalue,
				self.model_dropout_keep_prob: 1.0})
			a = self.model_loss_action.eval(feed_dict={
				self.model_placeholder_eye: data_state_eye,
				self.model_placeholder_skel: data_state_skel,
				self.model_placeholder_action: target_action,
				self.model_dropout_keep_prob: 1.0})
			print 'Loss values'
			print '\tqvalue:', q
			print '\taction:', a

		self.model_train_qvalue.run(feed_dict={
			self.model_placeholder_eye: data_state_eye,
			self.model_placeholder_skel: data_state_skel,
			self.model_placeholder_qvalue: target_qvalue,
			self.model_dropout_keep_prob: self.dropout_keep_prob})
		self.model_train_action.run(feed_dict={
			self.model_placeholder_eye: data_state_eye,
			self.model_placeholder_skel: data_state_skel,
			self.model_placeholder_action: target_action,
			self.model_dropout_keep_prob: self.dropout_keep_prob})
	def eval_action(self, state_eye, state_skel, action_default=None):
		s_eye = np.array([state_eye])
		s_skel = np.array([state_skel])
		val = self.model_eval_action.eval(feed_dict={
			self.model_placeholder_eye: s_eye,
			self.model_placeholder_skel: s_skel,
			self.model_dropout_keep_prob: 1.0})
		a = [val[0][0:6].tolist(),val[0][6:12].tolist(),val[0][12]]
		if action_default is None:
			return a
		else:
			return add_action(action_default,a)
	def eval_qvalue(self, state_eye, state_skel):
		s_eye = np.array([state_eye])
		s_skel = np.array([state_skel])
		val = self.model_eval_qvalue.eval(feed_dict={
			self.model_placeholder_eye: s_eye,
			self.model_placeholder_skel: s_skel,
			self.model_dropout_keep_prob: 1.0})
		return val[0][0]
	def create_model(self):
		# Create session
		self.sess = tf.InteractiveSession()
		# Check dimensions
		w,h = self.controller.get_eye().get_image_size()
		#print flatten(self.controller.get_state())
		#print flatten(self.controller.get_action_default())
		d = len(flatten(self.controller.get_state()))
		a = len(flatten(self.controller.get_action_default()))
		# 
		state_eye = tf.placeholder(tf.float32, [None,w,h,1])
		state_skel = tf.placeholder(tf.float32, [None,d])
		qvalue = tf.placeholder(tf.float32, [None,1])
		action = tf.placeholder(tf.float32, [None,a])
		keep_prob = tf.placeholder(tf.float32)
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
		W_fc2 = weight_variable([256+d, 128])
		b_fc2 = bias_variable([128])
		h_comb1 = tf.concat(1, [h_fc1, state_skel])
		h_fc2 = tf.nn.relu(tf.matmul(h_comb1, W_fc2) + b_fc2)
		h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
		#
		W_fc3_qvalue = weight_variable([128, 1])
		b_fc3_qvalue = bias_variable([1])
		h_fc3_qvalue = tf.matmul(h_fc2_drop, W_fc3_qvalue) + b_fc3_qvalue
		W_fc3_action = weight_variable([128, a])
		b_fc3_action = bias_variable([a])
		h_fc3_action = tf.matmul(h_fc2_drop, W_fc3_action) + b_fc3_action
		# Optimizer
		loss_qvalue = tf.reduce_mean(tf.square(qvalue - h_fc3_qvalue))
		self.model_train_qvalue = tf.train.AdamOptimizer(1e-4).minimize(loss_qvalue)
		loss_action = tf.reduce_mean(tf.square(action - h_fc3_action))
		self.model_train_action = tf.train.AdamOptimizer(1e-4).minimize(loss_action)
		# Dropout
		self.model_dropout_keep_prob = keep_prob
		# Evalutation
		self.model_loss_qvalue = loss_qvalue
		self.model_loss_action = loss_action
		# Place holders
		self.model_placeholder_eye = state_eye
		self.model_placeholder_skel = state_skel
		self.model_placeholder_qvalue = qvalue
		self.model_placeholder_action = action
		# Evaultion
		self.model_eval_qvalue = h_fc3_qvalue
		self.model_eval_action = h_fc3_action
		# Initialize all variables
		self.sess.run(tf.initialize_all_variables())
		self.model_initialized = True
	def postprocess_replay_buffer(self):
		max_size = self.max_buffer_size
		cur_size = self.buffer_size
		if cur_size > max_size:
			del self.replay_buffer[0:cur_size-max_size]
			self.buffer_size = max_size
	def reward(self):
		return self.scene.score()
	def sample(self, sample_size=50):
		data = []
		pick_history = []
		num_data = 0
		if self.buffer_size < sample_size:
			sample_size = self.buffer_size
		while num_data < sample_size:
			pick = np.random.randint(self.buffer_size)
			if pick in pick_history:
				continue
			else:
				pick_history.append(pick)
			data.append(self.replay_buffer[pick])
			num_data += 1
		return data
	def get_controller(self):
		return 0