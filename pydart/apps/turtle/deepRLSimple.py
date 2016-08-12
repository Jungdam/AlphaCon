import deepRL
import basics
import pickle
import scene
import action as ac
from numpy.linalg import inv
import mmMath
import numpy as np

class DeepRLSimple(deepRL.DeepRLBase):
	def __init__(self, world, scene, nn, warmup_file=None):
		deepRL.DeepRLBase.__init__(self, warmup_file)
		self.replay_buffer['actor'] = deepRL.ReplayBuffer()
		self.replay_buffer['critic'] = deepRL.ReplayBuffer()
		self.world = world
		self.scene = scene
		self.nn = nn
		if warmup_file is not None:
			# Load warmup data
			print self.warmup_file, "is loading..."
			data = self.convert_warmup_file_to_buffer_data(self.warmup_file)
			self.replay_buffer['actor'].append(data,verbose=True)
			self.warmup_size = self.replay_buffer['actor'].size_accum
			cnt = 0
			# Train random policy
			print '--------- Train random policy ---------'
			num_train_data = 2*self.warmup_size
			while True:
				self.train_action(self.sample_size,check_qvalue=False)
				cnt += self.sample_size
				if cnt%1000 == 0:
					print cnt, ' data were trainned'
				if cnt>=num_train_data:
					break
			print '---------------------------------------'
	def convert_warmup_file_to_buffer_data(self, file_name):
		f = open(file_name, 'r')
		data = pickle.load(f)
		size = len(data)
		action_default = self.world.skel.controller.get_action_default()
		tuples = []
		for d in data:
			q_skel_init = d[0]
			q_skel_term = d[1]
			action = d[2]

			self.world.reset()
			self.scene.perturbate()

			self.world.skel.set_positions(q_skel_init)
			self.world.step(False,False)
			self.scene.update()
			state_sensor_init = self.sensor()
			state_skel_init = self.world.skel.controller.get_state()
			reward_init = self.reward()

			self.world.reset()

			self.world.skel.set_positions(q_skel_term)
			self.world.step(False,False)
			self.scene.update()
			state_sensor_term = self.sensor()
			state_skel_term = self.world.skel.controller.get_state()
			reward_term = self.reward()

			reward = reward_term - reward_init

			action_extra = ac.sub(action,action_default)
			action_extra_flat = basics.flatten(action_extra)

			t = [state_sensor_init, state_skel_init, action_extra_flat, [reward], state_sensor_term, state_skel_term]
			if basics.check_valid_data(t):
				tuples.append(t)
		self.world.reset()
		return tuples
	def init_step(self):
		# initialize world and controller
		self.world.reset()
		# setup new environment
		self.scene.perturbate()
		self.scene.update()
	def sensor(self):
		R,p = mmMath.T2Rp(self.world.skel.body('trunk').T)
		return np.dot(inv(R),self.scene.get_pos()-p)
	def step(self, warming_up=False):
		# 
		state_sensor_init = self.sensor()
		state_skel_init = self.world.skel.controller.get_state()

		is_exploration = self.determine_exploration()

		# set new action parameter
		action_default = self.world.skel.controller.get_action_default()
		action_extra = []
		if warming_up:
			action_extra = ac.random([self.exprolation_noise]*ac.length())
			is_exploration = True
		else:
			action_extra = self.get_action(state_sensor_init, state_skel_init)
			if is_exploration:
				action_extra = ac.add(action_extra, ac.random([self.exprolation_noise]*ac.length()))
		action = ac.add(action_default, action_extra)
		self.world.skel.controller.add_action(action)
		action_extra_flat = basics.flatten(action_extra)
		while True:
			self.world.step()
			self.scene.update()
			if self.world.skel.controller.is_new_wingbeat():
				break
		reward = self.reward()
		state_sensor_term = self.sensor()
		state_skel_term = self.world.skel.controller.get_state()
		t = [state_sensor_init, state_skel_init, action_extra_flat, [reward], state_sensor_term, state_skel_term]
		buffer_name = 'critic'
		if is_exploration:
			buffer_name = 'actor'
		if basics.check_valid_data(t):
			return buffer_name, t
		else:
			return buffer_name, None
	def reward(self):
		return self.scene.score()
	def sample(self, buffer_name, idx):
		if not idx:
			return []
		data_state_sensor = []
		data_state_skel = []
		data_action = []
		data_reward = []
		data_state_sensor_prime = []
		data_state_skel_prime = []
		for i in idx:
			datum = self.replay_buffer[buffer_name].data[i]
			data_state_sensor.append(datum[0])
			data_state_skel.append(datum[1])
			data_action.append(datum[2])
			data_reward.append(datum[3])
			data_state_sensor_prime.append(datum[4])
			data_state_skel_prime.append(datum[5])
		return [ \
			np.array(data_state_sensor),np.array(data_state_skel),\
			np.array(data_action),\
			np.array(data_reward),\
			np.array(data_state_sensor_prime),np.array(data_state_skel_prime) ]
	def check_terminatation(self, state_sensor):
		return False
	def compute_target_qvalue(self, reward, state_sensor_prime, state_skel_prime):
		qvalue_prime = self.nn.eval_qvalue([state_sensor_prime, state_skel_prime])
		target_qvalue = reward + self.discount_factor*qvalue_prime
		return target_qvalue
	def loss_qvalue(self, sample_size=100, buffer_name='critic'):
		sample_idx = self.replay_buffer[buffer_name].sample_idx(sample_size)
		data = self.sample(buffer_name, sample_idx)
		q = 0.0
		if data:
			data_state_sensor = data[0]
			data_state_skel = data[1]
			data_reward = data[3]
			data_state_sensor_prime = data[4]
			data_state_skel_prime = data[5]
			target_qvalue = self.compute_target_qvalue(data_reward, data_state_sensor_prime, data_state_skel_prime)
			q = self.nn.loss_qvalue([data_state_sensor,data_state_skel,target_qvalue])
		return q
	def loss_action(self, sample_size=100, buffer_name='critic'):
		sample_idx = self.replay_buffer[buffer_name].sample_idx(sample_size)
		data = self.sample(buffer_name, sample_idx)
		a = 0.0
		if data:
			data_state_sensor = data[0]
			data_state_skel = data[1]
			data_action = data[2]
			target_action = data_action
			a = self.nn.loss_action([data_state_sensor,data_state_skel,target_action])
		return a
	def print_loss(self, sample_size=100):
		q = self.loss_qvalue(sample_size)
		a = self.loss_action(sample_size)
		print 'Loss values: ', 'qvalue:', q, 'action:', a
	def train_qvalue(self, sample_size):
		sample_idx = self.replay_buffer['critic'].sample_idx(sample_size)
		data = self.sample('critic', sample_idx)
		if data:
			data_state_sensor = data[0]
			data_state_skel = data[1]
			data_reward = data[3]
			data_state_sensor_prime = data[4]
			data_state_skel_prime = data[5]
			target_qvalue = self.compute_target_qvalue(data_reward, data_state_sensor_prime, data_state_skel_prime)
			self.nn.train_qvalue([data_state_sensor,data_state_skel,target_qvalue])
	def train_action(self, sample_size, check_qvalue=True):
		sample_idx = self.replay_buffer['actor'].sample_idx(sample_size)
		data = self.sample('actor', sample_idx)
		if data:
			data_state_sensor = data[0]
			data_state_skel = data[1]
			data_action = data[2]
			data_reward = data[3]
			data_state_sensor_prime = data[4]
			data_state_skel_prime = data[5]
			if check_qvalue:
				train_state_sensor = []
				train_state_skel = []
				train_target_action = []
				qvalue = self.nn.eval_qvalue([data_state_sensor, data_state_skel])
				target_qvalue = self.compute_target_qvalue(data_reward, data_state_sensor_prime, data_state_skel_prime)
				for i in xrange(len(data)):
					if target_qvalue[i][0] > qvalue[i][0]:
						train_state_sensor.append(data_state_sensor[i])
						train_state_skel.append(data_state_skel[i])
						train_target_action.append(data_action[i])
				if train_state_sensor:
					self.nn.train_action([train_state_sensor,train_state_skel,train_target_action])
			else:
				self.nn.train_action([data_state_sensor,data_state_skel,data_action])

	def train(self):
		self.train_qvalue(self.sample_size)
		self.train_action(self.sample_size)
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