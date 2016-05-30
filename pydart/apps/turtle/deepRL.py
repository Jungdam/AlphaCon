from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import warnings
import scene
import math
import action as ac
import pickle

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

class DeepRLBase:
	__metaclass__ = ABCMeta
	def __init__(self):
		self.replay_buffer = []
		self.buffer_size = 0
		self.buffer_size_accum = 0
		self.warmup_size = 10000
		self.max_buffer_size = 50000
		self.max_data_gen = 500000
		self.sample_size = 100
		self.discount_factor = 0.95
		self.init_exprolation_prob = 0.25
	def get_max_data_generation(self):
		return self.max_data_gen
	def get_buffer_size_accumulated(self):
		return self.buffer_size_accum
	#
	# This function initialize the environment to be simulated
	#
	@abstractmethod
	def init_step(self):
		raise NotImplementedError("Must override")
	# If the simulation result is valid,
	# 	this function returns a tuple [s,a,r,s'],
	# 	or returns None
	@abstractmethod
	def step(self):
		raise NotImplementedError("Must override")
	#
	# This function returns an immediate reward
	#
	@abstractmethod
	def reward(self):
		raise NotImplementedError("Must override")
	#
	# 
	#
	@abstractmethod
	def update_model(self, data):
		raise NotImplementedError("Must override")
	def postprocess_replay_buffer(self):
		max_size = self.max_buffer_size
		cur_size = self.buffer_size
		if cur_size > max_size:
			del self.replay_buffer[0:cur_size-max_size]
			self.buffer_size = max_size
	@abstractmethod
	def sample(self, idx):
		raise NotImplementedError("Must override")
	def sample_idx(self, sample_size=None):
		pick_history = []
		num_data = 0
		if sample_size is None:
			sample_size = self.sample_size
		if self.buffer_size < sample_size:
			sample_size = self.buffer_size
		while num_data < sample_size:
			pick = np.random.randint(self.buffer_size)
			if pick in pick_history:
				continue
			else:
				pick_history.append(pick)
			num_data += 1
		return pick_history
	def get_exprolation_prob(self):
		sigma_inv = 2.5
		return self.init_exprolation_prob * \
			math.exp(-sigma_inv*float(self.buffer_size_accum)/float(self.max_data_gen))
	def is_warming_up(self):
		return self.buffer_size_accum < self.warmup_size
	def is_finished_trainning(self):
		return self.buffer_size_accum >= self.max_data_gen
	def run(self, max_episode=100, max_iter=100, batch_train_iter=1):
		# Start trainning
		for i in xrange(max_episode):
			self.init_step()
			# Generate trainning tuples
			is_warming_up_start = self.is_warming_up()
			for j in xrange(max_iter):
				# print '\r', j, 'th iteration'
				t = self.step(self.get_exprolation_prob(),self.is_warming_up())
				# print t is None
				if t is None:
					self.init_step()
				else:
					self.replay_buffer.append(t)
					self.buffer_size += 1
					self.buffer_size_accum += 1
			is_warming_up_end = self.is_warming_up()
			if is_warming_up_start != is_warming_up_end:
				self.save_replay_buffer('warming_up_db.txt')
			if not self.is_warming_up():
				self.postprocess_replay_buffer()
				# Update the network
				for k in xrange(batch_train_iter):
					data = self.sample(self.sample_idx())
					self.update_model(data)
			print '[ ', i, 'th episode ]',\
				'buffer_size:',self.buffer_size,\
				'buffer_acum:',self.buffer_size_accum,\
				'warmup:', self.is_warming_up(),
			self.print_loss()
	def reset(self):
		del self.replay_buffer[:]
		self.buffer_size = 0
		self.buffer_size_accum = 0
	def save_replay_buffer(self, file_name):
		f = open(file_name, 'w')
		pickle.dump(self.replay_buffer, f)
		f.close()
		print '[Replay Buffer]', self.buffer_size, 'data are saved:', file_name
	def load_replay_buffer(self, file_name, append=False):
		f = open(file_name, 'r')
		data = pickle.load(f)
		if not append:
			self.reset()
		self.replay_buffer += data
		size = len(self.replay_buffer)
		self.buffer_size += size
		self.buffer_size_accum += size
		print '[Replay Buffer]', size, 'data are loaded:', file_name
class DeepRL(DeepRLBase):
	def __init__(self, world, skel, scene, nn):
		DeepRLBase.__init__(self)
		self.world = world
		self.skel = skel
		self.controller = skel.controller
		self.scene = scene
		self.nn = nn
	def init_step(self):
		# initialize world and controller
		self.world.reset()
		self.controller.reset()
		# set new environment
		self.scene.perturbate()
		self.scene.update(self.skel.body('trunk').T)
	def step(self, sigma, full_random=False):
		eye = self.controller.get_eye()
		# 
		state_eye_init = eye.get_image(self.skel.body('trunk').T)
		state_skel_init = self.controller.get_state()
		reward_init = self.reward()
		if self.check_terminatation(state_eye_init):
			return None
		# set new action parameter
		action_default = self.controller.get_action_default()
		action_extra = []
		if full_random:
			action_extra = ac.zero()
		else:
			action_extra = self.get_action(state_eye_init, state_skel_init)
		action_extra = ac.random([sigma]*ac.length(), action_extra)
		action = ac.add(action_default, action_extra)
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
		return [state_eye_init, state_skel_init, action_extra_flat, [reward], state_eye_term, state_skel_term]
	def reward(self):
		return self.scene.score()
	def sample(self, idx):
		data_state_eye = []
		data_state_skel = []
		data_action = []
		data_reward = []
		data_state_eye_prime = []
		data_state_skel_prime = []
		for i in idx:
			episode = self.replay_buffer[i]
			data_state_eye.append(episode[0])
			data_state_skel.append(episode[1])
			data_action.append(episode[2])
			data_reward.append(episode[3])
			data_state_eye_prime.append(episode[4])
			data_state_skel_prime.append(episode[5])
		return [ \
			np.array(data_state_eye),np.array(data_state_skel),\
			np.array(data_action),\
			np.array(data_reward),\
			np.array(data_state_eye_prime),np.array(data_state_skel_prime) ]
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
	def compute_target_value(self, data):
		data_state_eye_prime = data[0]
		data_state_skel_prime = data[1]
		data_action = data[2]
		data_reward = data[3]

		q = self.nn.eval_qvalue([data_state_eye_prime, data_state_skel_prime])

		target_qvalue = data_reward + self.discount_factor*q
		target_action = data_action

		return target_qvalue, target_action
	def print_loss(self, sample_size=200):
		data = self.sample(self.sample_idx(sample_size))
		data_state_eye = data[0]
		data_state_skel = data[1]
		data_action = data[2]
		data_reward = data[3]
		data_state_eye_prime = data[4]
		data_state_skel_prime = data[5]

		target_qvalue, target_action = \
			self.compute_target_value([
				data_state_eye_prime,
				data_state_skel_prime,
				data_action,
				data_reward])

		q = self.nn.loss_qvalue([data_state_eye,data_state_skel,target_qvalue])
		a = self.nn.loss_action([data_state_eye,data_state_skel,target_action])

		print 'Loss values: ', 'qvalue:', q, 'action:', a
	def update_model(self, data, print_loss=False):
		data_state_eye = data[0]
		data_state_skel = data[1]
		data_action = data[2]
		data_reward = data[3]
		data_state_eye_prime = data[4]
		data_state_skel_prime = data[5]

		target_qvalue, target_action = \
			self.compute_target_value([
				data_state_eye_prime,
				data_state_skel_prime,
				data_action,
				data_reward])

		self.nn.train([ \
			data_state_eye,data_state_skel,
			target_qvalue,target_action])
	def get_action(self, state_eye, state_skel, action_default=None):
		s_eye = np.array([state_eye])
		s_skel = np.array([state_skel])
		val = self.nn.eval_action([s_eye,s_skel])
		a = [val[0][0:6].tolist(),val[0][6:12].tolist(),val[0][12]]
		if action_default is None:
			return a
		else:
			return ac.add(action_default,a)
	def get_qvalue(self, state_eye, state_skel):
		s_eye = np.array([state_eye])
		s_skel = np.array([state_skel])
		val = self.nn.eval_qvalue([s_eye,s_skel])
		return val[0][0]