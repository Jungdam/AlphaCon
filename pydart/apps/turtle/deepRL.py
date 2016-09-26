from abc import ABCMeta, abstractmethod
import tensorflow as tf
import basics
import numpy as np

class ReplayBuffer:
	def __init__(self, max_size=100000):
		self.data = []
		self.size = 0
		self.size_accum = 0
		self.max_size = max_size
	def append(self, data, apply_pproces=True, verbose=False):
		num_valid = 0
		num_data = len(data)
		for datum in data:
			if not basics.check_valid_data(datum):
				print '[ReplayBuffer] Invalid data(NaN) was given to the buffer'
				continue
			self.data.append(datum)
			self.size += 1
			self.size_accum += 1
			num_valid += 1
		if apply_pproces:
			if self.size > self.max_size:
				del self.data[0:(self.size-self.max_size)]
				self.size = self.max_size
		if verbose:
			print '[Replay Buffer]', num_valid, 'data are added among', num_data, 'data'
			print '[Replay Buffer]', 'size: ', self.size, 'size_accum:', self.size_accum, 'max_size:', self.max_size
	def reset(self):
		del self.data[:]
		self.size = 0
		self.size_accum = 0
	def sample_idx(self, sample_size=100):
		pick_history = []
		num_picked_data = 0
		if self.size < sample_size:
			sample_size = self.size
		while num_picked_data < sample_size:
			pick = np.random.randint(self.size)
			if pick in pick_history:
				continue
			else:
				pick_history.append(pick)
				num_picked_data += 1
		return pick_history

class DeepRLBase:
	__metaclass__ = ABCMeta
	def __init__(self, warmup_file=None):
		self.replay_buffer = {}
		self.warmup_size = 50000
		self.max_data_gen = 1000000
		self.sample_size = 50
		self.discount_factor = 0.99
		self.exp_prob_default = 0.25
		self.exp_noise_default = 0.1
		self.warmup_file = warmup_file
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
	#
	@abstractmethod
	def train(self):
		raise NotImplementedError("Must override")
	@abstractmethod
	def sample(self, buffer_name, idx):
		raise NotImplementedError("Must override")
	@abstractmethod
	def convert_warmup_file_to_buffer_data(self, file_name):
		raise NotImplementedError("Must override")
	def num_data_gen(self):
		cnt = 0
		for i in self.replay_buffer.keys():
			cnt += self.replay_buffer[i].size_accum
		return cnt
	def determine_exploration(self):
		return np.random.uniform(0.0,1.0) < self.exp_prob_default
	def get_exploration_noise(self):
		num_data = self.num_data_gen()
		max_data = self.max_data_gen
		pivot = 0.25*max_data
		if num_data < pivot:
			return self.exp_noise_default
		else:
			r = (num_data - pivot)/max_data
			return self.exp_noise_default * (1.0-r) * (1.0-r)
	def is_warming_up(self):
		return self.num_data_gen() < self.warmup_size
	def is_finished_trainning(self):
		return self.num_data_gen() >= self.max_data_gen
	def run(self, max_episode=64, max_iter_per_episode=32, verbose=True):
		if self.is_finished_trainning():
			return
		for i in xrange(max_episode):
			self.init_step()
			# Generate trainning tuples
			is_warming_up_start = self.is_warming_up()
			for j in xrange(max_iter_per_episode):
				buffer_name, datum = self.step(self.is_warming_up())
				# None means that something wrong(e.g. blowing up) happens.
				if datum is None:
					self.init_step()
				else:
					self.replay_buffer[buffer_name].append([datum])
				# print i, j, buffer_name
			is_warming_up_end = self.is_warming_up()
			# # Save warming up data if necessary
			# if is_warming_up_start != is_warming_up_end:
			# 	self.save_replay_buffer('__warming_up_db.txt')
			# 	if self.warmup_file is None:
			# 		cnt = 0
			# 		while True:
			# 			self.update_model()
			# 			self.print_loss()
			# 			cnt += self.sample_size
			# 			if cnt>=2*self.buffer_size:
			# 				break
			# Train network
			if not self.is_warming_up():
				self.train()
			# Print statistics
			if verbose:
				print '[ ', i, 'th episode ]', ' warmup: ', self.is_warming_up(), 
				for buffer_name in self.replay_buffer.keys():
					print '[', buffer_name,
					print self.replay_buffer[buffer_name].size,
					print self.replay_buffer[buffer_name].size_accum, ']',
				if not self.is_warming_up():
					self.print_loss()
				else:
					print ' '
	def save_replay_buffer(self, file_name):
		f = open(file_name, 'w')
		pickle.dump(self.replay_buffer, f)
		f.close()
		print '[Replay Buffer]', self.buffer_size, 'data are saved:', file_name

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

