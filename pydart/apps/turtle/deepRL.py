from abc import ABCMeta, abstractmethod
import tensorflow as tf
import numpy as np
import warnings
import scene
import math
import action as ac
import pickle
from numpy.linalg import inv
import mmMath

def flatten(l):
	return list(_flatten_(l))

def _flatten_(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in _flatten_(*x):
                yield y
        else:
            yield x

class DeepRLBase:
	__metaclass__ = ABCMeta
	def __init__(self, warmup_file=None):
		self.replay_buffer = []
		self.buffer_size = 0
		self.buffer_size_accum = 0
		self.warmup_size = 45000
		self.max_buffer_size = 50000
		self.max_data_gen = 500000
		self.sample_size = 100
		self.discount_factor = 0.99
		self.init_exprolation_prob = 0.1
		self.warmup_file = warmup_file
		if self.warmup_file is not None:
			print self.warmup_file, "is loading..."
			data = self.convert_warmup_file_to_buffer_data(self.warmup_file)
			self.add_to_replay_buffer(data)
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
	@abstractmethod
	def sample(self, idx):
		raise NotImplementedError("Must override")
	@abstractmethod
	def convert_warmup_file_to_buffer_data(self, file_name):
		raise NotImplementedError("Must override")

	def postprocess_replay_buffer(self):
		max_size = self.max_buffer_size
		cur_size = self.buffer_size
		if cur_size > max_size:
			del self.replay_buffer[0:cur_size-max_size]
			self.buffer_size = max_size
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
		if self.is_warming_up():
			return self.init_exprolation_prob
		else:
			sigma_inv = 1.0
			return 1.0*self.init_exprolation_prob * \
				math.exp(-sigma_inv*float(self.buffer_size_accum)/float(self.max_data_gen))
	def is_warming_up(self):
		return self.buffer_size_accum < self.warmup_size
	def is_finished_trainning(self):
		return self.buffer_size_accum >= self.max_data_gen
	def run(self, max_episode=100, max_iter_per_episode=100, verbose=True):
		for i in xrange(max_episode):
			self.init_step()
			# Generate trainning tuples
			is_warming_up_start = self.is_warming_up()
			for j in xrange(max_iter_per_episode):
				# print '\r', j, 'th iteration'
				t = self.step(self.get_exprolation_prob(),self.is_warming_up())
				# print t is None
				if t is None:
					self.init_step()
				else:
					self.replay_buffer.append(t)
					self.buffer_size += 1
					self.buffer_size_accum += 1
			# Save warming up data if necessary
			is_warming_up_end = self.is_warming_up()
			if is_warming_up_start != is_warming_up_end:
				self.save_replay_buffer('__warming_up_db.txt')
				if self.warmup_file is None:
					cnt = 0
					while True:
						data = self.sample(self.sample_idx())
						self.update_model(data)
						self.print_loss()
						cnt += self.sample_size
						if cnt>=2*self.buffer_size:
							break
			# Train network
			if not self.is_warming_up():
				data = self.sample(self.sample_idx())
				self.update_model(data)
			# Print statistics
			if verbose:
				print '[ ', i, 'th episode ]',\
					'buffer_size:',self.buffer_size,\
					'buffer_acum:',self.buffer_size_accum,\
					'warmup:', self.is_warming_up(),
				if not self.is_warming_up():
					self.print_loss()
				else:
					print ' '
		if not self.is_warming_up():
			self.postprocess_replay_buffer()
	def reset(self):
		del self.replay_buffer[:]
		self.buffer_size = 0
		self.buffer_size_accum = 0
	def save_replay_buffer(self, file_name):
		f = open(file_name, 'w')
		pickle.dump(self.replay_buffer, f)
		f.close()
		print '[Replay Buffer]', self.buffer_size, 'data are saved:', file_name
	def add_to_replay_buffer(self, data, append=False):
		size = len(data)
		self.replay_buffer += data
		self.buffer_size += size
		self.buffer_size_accum += size
		print '[Replay Buffer]', size, 'data are added:'

# class DeepRL(DeepRLBase):
# 	def __init__(self, world, skel, scene, nn, warmup_file=None):
# 		DeepRLBase.__init__(self, warmup_file)
# 		self.world = world
# 		self.skel = skel
# 		self.controller = skel.controller
# 		self.scene = scene
# 		self.nn = nn
# 		# idx = []
# 		# for i in xrange(self.buffer_size):
# 		# 	data = self.replay_buffer[i]
# 		# 	reward = data[3]
# 		# 	state_eye_term = data[4]
# 		# 	if self.check_terminatation(state_eye_term) and reward[0]==0:
# 		# 		idx.append(i)
# 	def init_step(self):
# 		# initialize world and controller
# 		self.world.reset()
# 		self.controller.reset()
# 		# set new environment
# 		self.scene.perturbate()
# 		self.scene.update()
# 	def step(self, sigma, full_random=False):
# 		eye = self.controller.get_eye()
# 		# 
# 		state_eye_init = eye.get_image(self.skel.body('trunk').T)
# 		state_skel_init = self.controller.get_state()
# 		reward_l_init, reward_a_init = self.reward()
# 		if self.check_terminatation(state_eye_init):
# 			return None
# 		# set new action parameter
# 		action_default = self.controller.get_action_default()
# 		action_extra = []
# 		if full_random:
# 			action_extra = ac.zero()
# 		else:
# 			action_extra = self.get_action(state_eye_init, state_skel_init)
# 		action_extra = ac.random([sigma]*ac.length(), action_extra)
# 		action = ac.add(action_default, action_extra)
# 		# print action
# 		self.controller.add_action(action)
# 		action_extra_flat = flatten(action_extra)
# 		while True:
# 			self.world.step()
# 			self.scene.update()
# 			if self.controller.is_new_wingbeat():
# 				break
# 		state_eye_term = eye.get_image(self.skel.body('trunk').T)
# 		state_skel_term = self.controller.get_state()
# 		reward_l_term, reward_a_term = self.reward()
# 		reward = 0.1* ((reward_l_term - reward_l_init) + (reward_a_term - reward_a_init))
# 		if self.check_terminatation(state_eye_term) and reward==0:
# 			return None
# 		return [state_eye_init, state_skel_init, action_extra_flat, [reward], state_eye_term, state_skel_term]
# 	def reward(self):
# 		return self.scene.score()
# 	def sample(self, idx):
# 		data_state_eye = []
# 		data_state_skel = []
# 		data_action = []
# 		data_reward = []
# 		data_state_eye_prime = []
# 		data_state_skel_prime = []
# 		for i in idx:
# 			episode = self.replay_buffer[i]
# 			data_state_eye.append(episode[0])
# 			data_state_skel.append(episode[1])
# 			data_action.append(episode[2])
# 			data_reward.append(episode[3])
# 			data_state_eye_prime.append(episode[4])
# 			data_state_skel_prime.append(episode[5])
# 		return [ \
# 			np.array(data_state_eye),np.array(data_state_skel),\
# 			np.array(data_action),\
# 			np.array(data_reward),\
# 			np.array(data_state_eye_prime),np.array(data_state_skel_prime) ]
# 	def check_terminatation(self, state_eye):
# 		# No valid depth image
# 		threshold = 0.99
# 		w,h,c = state_eye.shape
# 		term_eye = True
# 		for i in xrange(w):
# 			if term_eye is False:
# 				break
# 			for j in xrange(h):
# 				val = state_eye[i,j,0]
# 				if val <= threshold:
# 					term_eye = False
# 					break
# 		if term_eye:
# 			return True
# 		# Passed every termination conditions
# 		return False
# 	def compute_target_value(self, data):
# 		data_state_eye_prime = data[0]
# 		data_state_skel_prime = data[1]
# 		data_action = data[2]
# 		data_reward = data[3]

# 		q = self.nn.eval_qvalue([data_state_eye_prime, data_state_skel_prime])

# 		target_qvalue = data_reward + self.discount_factor*q
# 		target_action = data_action
# 		return target_qvalue, target_action
# 	def print_loss(self, sample_size=100):
# 		data = self.sample(self.sample_idx(sample_size))
# 		data_state_eye = data[0]
# 		data_state_skel = data[1]
# 		data_action = data[2]
# 		data_reward = data[3]
# 		data_state_eye_prime = data[4]
# 		data_state_skel_prime = data[5]

# 		target_qvalue, target_action = \
# 			self.compute_target_value([
# 				data_state_eye_prime,
# 				data_state_skel_prime,
# 				data_action,
# 				data_reward])

# 		q = self.nn.loss_qvalue([data_state_eye,data_state_skel,target_qvalue])
# 		a = self.nn.loss_action([data_state_eye,data_state_skel,target_action])

# 		print 'Loss values: ', \
# 			'qvalue:', q, \
# 			'action:', a, \
# 			'exp:', np.array(self.get_exprolation_prob())
# 	def update_model(self, data, print_loss=False):
# 		data_state_eye = data[0]
# 		data_state_skel = data[1]
# 		data_action = data[2]
# 		data_reward = data[3]
# 		data_state_eye_prime = data[4]
# 		data_state_skel_prime = data[5]
# 		target_qvalue, target_action = \
# 			self.compute_target_value([
# 				data_state_eye_prime,
# 				data_state_skel_prime,
# 				data_action,
# 				data_reward])
# 		self.nn.train_qvalue([data_state_eye,data_state_skel,target_qvalue])
# 		# target_qvalue, target_action = \
# 		# 	self.compute_target_value([
# 		# 		data_state_eye_prime,
# 		# 		data_state_skel_prime,
# 		# 		data_action,
# 		# 		data_reward])
# 		qvalue = self.nn.eval_qvalue([data_state_eye, data_state_skel])
# 		dse = []
# 		dss = []
# 		ta = []
# 		num_data = len(data_state_eye)
# 		for i in xrange(num_data):
# 			if True or target_qvalue[i][0] > qvalue[i][0]:
# 				dse.append(data_state_eye[i])
# 				dss.append(data_state_skel[i])
# 				ta.append(target_action[i])
# 		if len(ta)>0:
# 			self.nn.train_action([dse,dss,ta])
# 		# self.nn.train([ \
# 		# 	data_state_eye,data_state_skel,
# 		# 	target_qvalue,target_action])
# 	def get_action(self, state_eye, state_skel, action_default=None):
# 		s_eye = np.array([state_eye])
# 		s_skel = np.array([state_skel])
# 		val = self.nn.eval_action([s_eye,s_skel])
# 		a = [val[0][0:6].tolist(),val[0][6:12].tolist(),val[0][12]]
# 		if action_default is None:
# 			return a
# 		else:
# 			return ac.add(action_default,a)
# 	def get_qvalue(self, state_eye, state_skel):
# 		s_eye = np.array([state_eye])
# 		s_skel = np.array([state_skel])
# 		val = self.nn.eval_qvalue([s_eye,s_skel])
# 		return val[0][0]
# 	def save_replay_test(self, file_name):
# 		idx = self.sample_idx(1000)
# 		f = open(file_name, 'w')
# 		for i in idx:
# 			data = self.replay_buffer[i]
# 			s = "R: "+str(data[3])+"\n"\
# 				"A: "+str(['{:.3f}'.format(i) for i in data[2]])+"\n"+\
# 				"S0: "+str(['{:.3f}'.format(i) for i in data[1]])+"\n"+\
# 				"S1: "+str(['{:.3f}'.format(i) for i in data[5]])+"\n"
# 			f.write(s)
# 		f.close()

class DeepRLSimple(DeepRLBase):
	def __init__(self, world, scene, nn, warmup_file=None):
		self.world = world
		self.scene = scene
		self.nn = nn
		DeepRLBase.__init__(self, warmup_file)
		if warmup_file is not None:
			cnt = 0
			while True:
				data = self.sample(self.sample_idx())
				self.update_model(data)
				self.print_loss()
				cnt += self.sample_size
				if cnt>=0.1*self.buffer_size:
					break;
	def convert_warmup_file_to_buffer_data(self, file_name):
		f = open(file_name, 'r')
		data = pickle.load(f)
		size = len(data)
		action_default = self.world.skel.controller.get_action_default()
		tuples = []
		for d in data:
			q_skel_init = d[0]
			q_skel_term = d[1]

			self.world.reset()
			self.scene.perturbate()
			self.scene.update()

			self.world.skel.set_positions(q_skel_init)
			self.world.step(False)
			self.scene.update()
			state_sensor_init = self.sensor()
			state_skel_init = self.world.skel.controller.get_state()
			reward_init = self.reward()

			if reward_init > 0.0:
				continue

			self.world.reset()
			self.world.skel.controller.reset()

			self.world.skel.set_positions(q_skel_term)
			self.world.step(False)
			self.scene.update()
			state_sensor_term = self.sensor()
			state_skel_term = self.world.skel.controller.get_state()
			reward_term = self.reward()

			reward = reward_term - reward_init

			action = d[2]
			action_extra = ac.sub(action,action_default)
			action_extra_flat = flatten(action_extra)
			tuples.append([state_sensor_init, state_skel_init, action_extra_flat, [reward], state_sensor_term, state_skel_term])

		self.world.reset()
		return tuples
	def init_step(self):
		# initialize world and controller
		self.world.reset()
		# set new environment
		self.scene.perturbate()
		self.scene.update()
	def sensor(self):
		R,p = mmMath.T2Rp(self.world.skel.body('trunk').T)
		return np.dot(inv(R),self.scene.get_pos()-p)
	def step(self, sigma, full_random=False):
		# 
		state_sensor_init = self.sensor()
		state_skel_init = self.world.skel.controller.get_state()
		reward_init = self.reward()
		if reward_init > 0.0:
			return None
		# set new action parameter
		action_default = self.world.skel.controller.get_action_default()
		action_random = ac.random([sigma]*ac.length())
		action_extra = []
		if full_random:
			action_extra = action_random
		else:
			action_policy = self.get_action(state_sensor_init, state_skel_init)
			action_extra = ac.add(action_policy, action_random)
		action = ac.add(action_default, action_extra)
		# print action
		self.world.skel.controller.add_action(action)
		action_extra_flat = flatten(action_extra)
		while True:
			self.world.step()
			self.scene.update()
			if self.world.skel.controller.is_new_wingbeat():
				break
		state_sensor_term = self.sensor()
		state_skel_term = self.world.skel.controller.get_state()
		reward_term = self.reward()
		reward = reward_term - reward_init
		# print reward_l_init, reward_l_term
		# print reward_a_init, reward_a_term
		# print reward, '\n'
		return [state_sensor_init, state_skel_init, action_extra_flat, [reward], state_sensor_term, state_skel_term]
	def reward(self):
		return self.scene.score()
	def sample(self, idx):
		data_state_sensor = []
		data_state_skel = []
		data_action = []
		data_reward = []
		data_state_sensor_prime = []
		data_state_skel_prime = []
		for i in idx:
			episode = self.replay_buffer[i]
			data_state_sensor.append(episode[0])
			data_state_skel.append(episode[1])
			data_action.append(episode[2])
			data_reward.append(episode[3])
			data_state_sensor_prime.append(episode[4])
			data_state_skel_prime.append(episode[5])
		return [ \
			np.array(data_state_sensor),np.array(data_state_skel),\
			np.array(data_action),\
			np.array(data_reward),\
			np.array(data_state_sensor_prime),np.array(data_state_skel_prime) ]
	def check_terminatation(self, state_sensor):
		return False
	def compute_target_value(self, data):
		data_state_sensor_prime = data[0]
		data_state_skel_prime = data[1]
		data_action = data[2]
		data_reward = data[3]

		q = self.nn.eval_qvalue([data_state_sensor_prime, data_state_skel_prime])

		target_qvalue = data_reward + self.discount_factor*q
		target_action = data_action
		return target_qvalue, target_action
	def print_loss(self, sample_size=100):
		data = self.sample(self.sample_idx(sample_size))
		data_state_sensor = data[0]
		data_state_skel = data[1]
		data_action = data[2]
		data_reward = data[3]
		data_state_sensor_prime = data[4]
		data_state_skel_prime = data[5]

		target_qvalue, target_action = \
			self.compute_target_value([
				data_state_sensor_prime,
				data_state_skel_prime,
				data_action,
				data_reward])

		q = self.nn.loss_qvalue([data_state_sensor,data_state_skel,target_qvalue])
		a = self.nn.loss_action([data_state_sensor,data_state_skel,target_action])

		print 'Loss values: ', \
			'qvalue:', q, \
			'action:', a, \
			'exp:', np.array(self.get_exprolation_prob())
	def update_model(self, data, print_loss=False):
		data_state_sensor = data[0]
		data_state_skel = data[1]
		data_action = data[2]
		data_reward = data[3]
		data_state_sensor_prime = data[4]
		data_state_skel_prime = data[5]
		
		target_qvalue, target_action = \
			self.compute_target_value([
				data_state_sensor_prime,
				data_state_skel_prime,
				data_action,
				data_reward])

		self.nn.train_qvalue([data_state_sensor,data_state_skel,target_qvalue])

		# self.nn.train([data_state_sensor,data_state_skel,target_qvalue,target_action])
		# return

		qvalue = self.nn.eval_qvalue([data_state_sensor, data_state_skel])
		train_state_sensor = []
		train_state_skel = []
		train_target_action = []
		train_target_qvalue = []
		num_data = len(data_state_sensor)
		for i in xrange(num_data):
			if False or target_qvalue[i][0] > qvalue[i][0]:
				train_state_sensor.append(data_state_sensor[i])
				train_state_skel.append(data_state_skel[i])
				train_target_action.append(target_action[i])
				train_target_qvalue.append(target_qvalue[i])
		if len(train_target_qvalue)>0:
			self.nn.train_action([train_state_sensor,train_state_skel,train_target_action])
	def get_action(self, state_sensor, state_skel, action_default=None):
		s_sensor = np.array([state_sensor])
		s_skel = np.array([state_skel])
		val = self.nn.eval_action([s_sensor,s_skel])
		a = [val[0][0:6].tolist(),val[0][6:12].tolist(),val[0][12]]
		if action_default is None:
			return a
		else:
			return ac.add(action_default,a)
	def get_qvalue(self, state_sensor, state_skel):
		s_sensor = np.array([state_sensor])
		s_skel = np.array([state_skel])
		val = self.nn.eval_qvalue([s_sensor,s_skel])
		return val[0][0]
	def save_replay_test(self, file_name):
		idx = self.sample_idx(1000)
		f = open(file_name, 'w')
		for i in idx:
			data = self.replay_buffer[i]
			s = "R: "+str(data[3])+"\n"\
				"A: "+str(['{:.3f}'.format(i) for i in data[2]])+"\n"+\
				"S0: "+str(['{:.3f}'.format(i) for i in data[1]])+"\n"+\
				"S1: "+str(['{:.3f}'.format(i) for i in data[5]])+"\n"
			f.write(s)
		f.close()