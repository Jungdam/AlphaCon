from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pydart
import controller
import gl_render
import numpy as np
import math
import basics
import deepRL
import nn
import tensorflow as tf
import mmMath
import action as ac
import pickle

np.set_printoptions(precision=3)
flag = {}
flag['Train'] = False
cnt_target_update = 0
max_target_update = 20
log_dir = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/tensorflow/log'
ckpt_dir = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/tensorflow/model/'
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle.skel'
warmup_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/warmup/0.15_10000_8.warmup'

pydart.init()
print('pydart initialization OK')

class Envi:
	def __init__(self, dt, skel_file, num_init_wingbeat=2):
		self.world = pydart.create_world(dt, skel_file)
		self.skel = self.world.skels[0]
		self.skel.controller = controller.Controller(self.world, self.skel)
		self.target_mu = np.array([0,1.0,7.0])
		self.target_sigma = np.array([1.0,1.0,2.0])
		self.target = np.random.normal(self.target_mu,self.target_sigma)
		while True:
		    if self.skel.controller.get_num_wingbeat() >= num_init_wingbeat:
		        self.world.push()
		        self.world.push()
		        self.set_random_state()
		        break;
		    self.world.step()
	def get_target_pos(self):
		return self.target
	def set_random_state(self):
		self.world.reset()
		self.skel.controller.reset()
		self.target = np.random.normal(self.target_mu,self.target_sigma)
		return
	def goodness(self):
		R,p = mmMath.T2Rp(self.skel.body('trunk').T)
		diff = p-self.target
		l = np.linalg.norm(diff)
		return math.exp(-0.5*l*l)
	def state(self):
		#
		# State of skeleton
		#
		state_skel = []
		# prepare for computing local coordinate of the trunk
		body_trunk = self.skel.body('trunk')
		R_trunk,p_trunk = mmMath.T2Rp(body_trunk.T)
		R_trunk_inv = np.linalg.inv(R_trunk)
		# trunk state 
		vel_trunk = body_trunk.world_com_velocity()
		state_skel.append(np.dot(R_trunk_inv,vel_trunk))
		# other bodies state
		bodies = []
		# bodies = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
		# bodies = ['left_arm', 'right_arm']
		for name in bodies:
			body = self.skel.body(name)
			l = body.world_com() - p_trunk
			v = body.world_com_velocity()
			state_skel.append(np.dot(R_trunk_inv,l))
			state_skel.append(np.dot(R_trunk_inv,v))
		state_skel = np.array(state_skel).flatten()
        #
		# State of sensor
		#
		state_sensor = np.dot(R_trunk_inv,self.get_target_pos()-p_trunk)
		return np.hstack([state_skel,state_sensor])
	def step_forward(self, apply_controller=True, apply_aero=True):
		self.world.step(apply_controller,apply_aero)
		return self.goodness()
	def step_forward_time(self, delta=1.0/10.0):
		elpased = 0.0
		while elpased < delta:
			self.world.step(True,True)
			elpased += dt
		return self.goodness()
	def step_forward_wingbeat(self, num_wingbeat=1):
		cnt_wingbeat = 0
		while cnt_wingbeat < num_wingbeat:
			self.world.step(True,True)
			if self.skel.controller.is_new_wingbeat():
				cnt_wingbeat += 1
		return self.goodness()
	def render(self):
		if self.world is not None:
			self.world.render()
		glColor3d(1.0, 0.0, 0.0)
		p = self.get_target_pos()
		glPushMatrix()
		glTranslated(p[0],p[1],p[2])
		glutSolidSphere(0.25, 10, 10)
		glPopMatrix()

class NN(nn.NNBase):
	def __init__(self, name):
		nn.NNBase.__init__(self, name)
		self.dropout_keep_prob = 1.0
		self.train_a = None
		self.train_q = None
		self.eval_q = None
		self.eval_a = None
		self.eval_q_copy = None
		self.eval_a_copy = None
		self.placeholder_state = None
		self.placeholder_target_qvalue = None
		self.placeholder_target_action = None
		self.placeholder_dropout_keep_prob = None
		self.loss_q = None
		self.loss_a = None
		self.learning_rate = 0.001
		self.writer = None
		self.merged = None
	def initialize(self, data, ckpt_file=None):
		tf.reset_default_graph()
		with self.graph.as_default():
			d = data[0]		# state dimension
			a = data[1]		# action dimension
			# 
			state = tf.placeholder(tf.float32, [None,d])
			target_qvalue = tf.placeholder(tf.float32, [None,1])
			target_action = tf.placeholder(tf.float32, [None,a])
			keep_prob = tf.placeholder(tf.float32)

			#
			# Network Definition
			#

			layer_1 = nn.Layer('layer_1',self.var, False, state, d, 32)
			layer_2 = nn.Layer('layer_2',self.var, False, layer_1.h, 32, 32)
			layer_3 = nn.Layer('layer_3',self.var, False, layer_2.h, 32, 16, 
				dropout_enabled=True, dropout_placeholder=keep_prob)
			layer_q = nn.Layer('layer_q',self.var, False, layer_3.h, 16, 1, None)
			layer_a = nn.Layer('layer_a',self.var, False, layer_3.h, 16, a, None)

			layer_1_copy = layer_1.copy(state)
			layer_2_copy = layer_2.copy(layer_1_copy.h)
			layer_3_copy = layer_3.copy(layer_2_copy.h)
			layer_q_copy = layer_q.copy(layer_3_copy.h)
			layer_a_copy = layer_a.copy(layer_3_copy.h)

			# self.var._print()

			h_qvalue = layer_q.h
			h_action = layer_a.h
			h_qvalue_copy = layer_q_copy.h
			h_action_copy = layer_a_copy.h

			# Optimizer
			loss_qvalue = tf.reduce_mean(tf.square(target_qvalue - h_qvalue))
			loss_action = tf.reduce_mean(tf.square(target_action - h_action))			

			# global_step = tf.Variable(0, trainable=False)
			# learning_rate = tf.train.exponential_decay(
			# 	self.learning_rate, global_step, 1000000, 0.96, staircase=True)

			# self.train_q = tf.train.AdamOptimizer(1e-3).minimize(loss_qvalue)
			# self.train_a = tf.train.AdamOptimizer(1e-3).minimize(loss_action)
			opt_q = tf.train.GradientDescentOptimizer(self.learning_rate)
			opt_a = tf.train.GradientDescentOptimizer(self.learning_rate)
			# opt_q = tf.train.AdamOptimizer(1e-4)
			# opt_a = tf.train.AdamOptimizer(1e-4)

			grads_and_vars_q = opt_q.compute_gradients(loss_qvalue)
			grads_and_vars_a = opt_a.compute_gradients(loss_action)

			capped_grads_and_vars_q = [\
				(tf.clip_by_norm(grad,10.0), var) if grad is not None \
				else (None, var) for grad, var in grads_and_vars_q]
			capped_grads_and_vars_a = [\
				(tf.clip_by_norm(grad,10.0), var) if grad is not None \
				else (None, var) for grad, var in grads_and_vars_a]
			
			# Trainning
			self.train_q = opt_q.apply_gradients(capped_grads_and_vars_q)
			self.train_a = opt_a.apply_gradients(capped_grads_and_vars_a)
			# Evaultion
			self.eval_q = h_qvalue
			self.eval_a = h_action
			self.eval_q_copy = h_qvalue_copy
			self.eval_a_copy = h_action_copy
			# Place holders
			self.placeholder_state = state
			self.placeholder_target_qvalue = target_qvalue
			self.placeholder_target_action = target_action
			self.placeholder_dropout_keep_prob = keep_prob
			# Loss
			self.loss_q = loss_qvalue
			self.loss_a = loss_action
			# Session
			self.sess = tf.Session(graph=self.graph)			
			# Saver
			self.saver = tf.train.Saver()
			success = self.restore(ckpt_file)
			if not success:
				print '[ NeuralNet ]', 'Variables are randomly initialzed'
				self.sess.run(tf.initialize_all_variables())
			# Summary
			self.merged = tf.merge_all_summaries()
			self.writer = tf.train.SummaryWriter(log_dir, self.graph)
			self.writer.flush()
			self.initialized = True
	def eval_qvalue(self, data, from_copy=False):
		if from_copy:
			return self.sess.run(self.eval_q_copy, feed_dict={
				self.placeholder_state: data[0]})
		else:
			return self.sess.run(self.eval_q, feed_dict={
				self.placeholder_state: data[0],
				self.placeholder_dropout_keep_prob: 1.0})
	def eval_action(self, data, from_copy=False):
		if from_copy:
			return self.sess.run(self.eval_a_copy, feed_dict={
				self.placeholder_state: data[0]})
		else:
			return self.sess.run(self.eval_a, feed_dict={
				self.placeholder_state: data[0],
				self.placeholder_dropout_keep_prob: 1.0})
	def loss_qvalue(self, data):
		val = self.sess.run(self.loss_q,feed_dict={
			self.placeholder_state: data[0],
			self.placeholder_target_qvalue: data[1],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def loss_action(self, data):
		val = self.sess.run(self.loss_a,feed_dict={
			self.placeholder_state: data[0],
			self.placeholder_target_action: data[1],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def train_qvalue(self, data):
		self.sess.run(self.train_q, feed_dict={
			self.placeholder_state: data[0],
			self.placeholder_target_qvalue: data[1],
			self.placeholder_dropout_keep_prob: self.dropout_keep_prob})
	def train_action(self, data):
		self.sess.run(self.train_a, feed_dict={
			self.placeholder_state: data[0],
			self.placeholder_target_action: data[1],
			self.placeholder_dropout_keep_prob: self.dropout_keep_prob})
	def save_variables(self):
		self.var.save(self.sess)

class DeepRL(deepRL.DeepRLBase):
	def __init__(self, envi, nn, warmup_file=None):
		deepRL.DeepRLBase.__init__(self, warmup_file)
		self.replay_buffer['actor'] = deepRL.ReplayBuffer(1000000)
		self.replay_buffer['critic'] = deepRL.ReplayBuffer(1000000)
		self.envi = envi
		self.nn = nn
		self.warmup_size = 10
		self.max_data_gen = 2000000
		self.sample_size = 32
		if warmup_file is None:
			print '[DeepRL]', 'generating warmup data ...'
			cnt = 0
			while True:
				self.init_step()
				buffer_name, datum = self.step(full_random=True)
				if datum is None:
					self.init_step()
				else:
					self.replay_buffer['actor'].append([datum])
					cnt += 1
				if cnt >= self.warmup_size:
					break
				if cnt%1000 == 0:
					print cnt, ' data were generated'
		else:
			print '[DeepRL]', 'loading warmup file ...'
			data = self.convert_warmup_file_to_buffer_data(self.warmup_file)
			self.replay_buffer['actor'].append(data,verbose=True)
			self.warmup_size = self.replay_buffer['actor'].size_accum
		self.envi.set_random_state()
		self.save_variables()
	def convert_warmup_file_to_buffer_data(self, file_name):
		f = open(file_name, 'r')
		data = pickle.load(f)
		size = len(data)
		action_default = self.envi.world.skel.controller.get_action_default()
		tuples = []
		cnt = 0
		for d in data:
			q_skel_init = d[0]
			q_skel_term = d[1]
			action = d[2]

			self.envi.set_random_state()

			self.envi.world.skel.set_positions(q_skel_init)
			reward_init = self.envi.step_forward(False,False)
			state_init = self.envi.state()

			self.envi.world.skel.set_positions(q_skel_term)
			reward_term = self.envi.step_forward(False,False)
			state_term = self.envi.state()

			reward = reward_term

			action_extra = ac.sub(action,action_default)
			action_extra_flat = np.array(basics.flatten(action_extra))

			t = [state_init, action_extra_flat, [reward], state_term]
			if basics.check_valid_data(t):
				tuples.append(t)
				cnt += 1
				if cnt%5000 == 0:
					print cnt, ' data were loaded'
		return tuples
	def init_step(self):
		self.envi.set_random_state()
	def step(self, full_random=False, force_buffer_name=None):
		buffer_name = 'critic'
		is_exploration = self.determine_exploration()

		state_init = self.envi.state()
		action = self.get_action(state_init)
		action_default = np.array(ac.flat(self.envi.world.skel.controller.get_action_default()))
		if full_random:
			mu = np.zeros(ac.length())
			sigma = 0.2*np.ones(ac.length())
			action = np.random.normal(mu, sigma)
			buffer_name = 'actor'
		elif is_exploration:
			mu = np.zeros(ac.length())
			sigma = 0.1*np.ones(ac.length())
			action += np.random.normal(mu, sigma)
			buffer_name = 'actor'
		action = action + action_default
		reward = self.envi.step_forward_wingbeat()
		state_term = self.envi.state()
		t = [state_init, action, [reward], state_term]
		# print state_init, action, reward
		if force_buffer_name is not None:
			buffer_name = force_buffer_name
		if basics.check_valid_data(t):
			return buffer_name, t
		else:
			return buffer_name, None
	def sample(self, buffer_name, idx):
		if not idx:
			return []
		data_state = []
		data_action = []
		data_reward = []
		data_state_prime = []
		for i in idx:
			datum = self.replay_buffer[buffer_name].data[i]
			data_state.append(datum[0])
			data_action.append(datum[1])
			data_reward.append(datum[2])
			data_state_prime.append(datum[3])
		return [ \
			np.array(data_state),\
			np.array(data_action),\
			np.array(data_reward),\
			np.array(data_state_prime)]
	def compute_target_qvalue(self, reward, state_prime):
		qvalue_prime = self.nn.eval_qvalue([state_prime], True)
		target_qvalue = reward + self.discount_factor*qvalue_prime
		return target_qvalue
	def train_qvalue(self, sample_size, iteration=10, verbose=False):
		sample_idx = self.replay_buffer['critic'].sample_idx(sample_size)
		data = self.sample('critic', sample_idx)
		if data:
			data_state = data[0]
			data_reward = data[2]
			data_state_prime = data[3]
			target_qvalue = self.compute_target_qvalue(data_reward, data_state_prime)
			for i in range(iteration):
				self.nn.train_qvalue([data_state,target_qvalue])
				if verbose:
					print self.nn.loss_qvalue([data_state,target_qvalue])
	def train_action(self, sample_size, check_qvalue=True, iteration=10, verbose=False):
		sample_idx = self.replay_buffer['actor'].sample_idx(sample_size)
		data = self.sample('actor', sample_idx)
		if data:
			data_state = data[0]
			data_action = data[1]
			data_reward = data[2]
			data_state_prime = data[3]
			if check_qvalue:
				train_state = []
				train_action = []
				qvalue = self.nn.eval_qvalue([data_state], True)
				target_qvalue = self.compute_target_qvalue(data_reward, data_state_prime)
				for i in xrange(len(qvalue)):
					if target_qvalue[i][0] > qvalue[i][0]:
						train_state.append(data_state[i])
						train_action.append(data_action[i])
				data_state = train_state
				data_action = train_action
			if data_state:
				for i in range(iteration):
					self.nn.train_action([data_state,data_action])
					if verbose:
						print self.nn.loss_action([data_state,data_action])
	def train(self):
		self.train_qvalue(self.sample_size)
		self.train_action(self.sample_size)
	def get_action(self, state):
		val = self.nn.eval_action([[state]], True)
		return val[0]
	def get_qvalue(self, state):
		val = self.nn.eval_qvalue([[state]], True)
		return val[0][0]
	def loss_qvalue(self, sample_size=100, buffer_name='critic'):
		sample_idx = self.replay_buffer[buffer_name].sample_idx(sample_size)
		data = self.sample(buffer_name, sample_idx)
		q = 0.0
		if data:
			data_state = data[0]
			data_reward = data[2]
			data_state_prime = data[3]
			target_qvalue = self.compute_target_qvalue(data_reward, data_state_prime)
			q = self.nn.loss_qvalue([data_state,target_qvalue])
		return q
	def loss_action(self, sample_size=100, buffer_name='critic'):
		sample_idx = self.replay_buffer[buffer_name].sample_idx(sample_size)
		data = self.sample(buffer_name, sample_idx)
		if data:
			data_state = data[0]
			data_action = data[1]
			return self.nn.loss_action([data_state, data_action])
		else:
			return 0.0
	def print_loss(self, sample_size=100):
		q = self.loss_qvalue(sample_size)
		a = self.loss_action(sample_size)
		print 'Loss values: ', 'qvalue:', q, 'action:', a
	def save_variables(self):
		self.nn.save_variables()

dt = 1.0/600.0
myEnvi = Envi(dt, skel_file)
myNN = NN('net_turtle')
myNN.initialize([len(myEnvi.state()),ac.length()])
# # myNN.initialize([len(myEnvi.state()),2],ckpt_dir)
myDeepRL = DeepRL(myEnvi, myNN, warmup_file)

def step_callback():
	return

def render_callback():
	gl_render.render_ground(color=[1.0,1.0,1.0],axis='y')
	myEnvi.render()
	# gl_render.render_ground(color=[1.0,1.0,1.0],axis='z')
	# myEnvi.detective.render(color=[0,0,1])
	# myEnvi.criminal.render(color=[1,0,0])
	if flag['Train']:
		global cnt_target_update
		myDeepRL.run(50, 20)
		if cnt_target_update>=max_target_update:
			myDeepRL.save_variables()
			cnt_target_update = 0
		else:
			cnt_target_update += 1

def keyboard_callback(key):
	if key == 'r':
		myEnvi.set_random_state()
	elif key == 'p':
		myEnvi.step_forward_time()
	elif key == 't':
		print myEnvi.state()
	elif key == ' ':
		elpased = 0.0
		while True:
			myEnvi.step_forward()
			if myEnvi.skel.controller.is_new_wingbeat():
				state = myEnvi.state()
				action = myDeepRL.get_action(state)
				qvalue = myDeepRL.get_qvalue(state)
				reward = myEnvi.goodness()
				print 'S:', state, 'A:', action, 'R:', reward, 'Q:', qvalue
				myEnvi.skel.controller.add_action(action, True)
			elpased += dt
			if elpased >= 0.1:
				break
	elif key == 'd':
		flag['Train'] = not flag['Train']
	elif key == 's':
		myNN.save(ckpt_dir)
	elif key == '0':
		sample_idx = myDeepRL.replay_buffer['critic'].sample_idx(10)
		data = myDeepRL.sample('critic', sample_idx)
		print data

pydart.glutgui.glutgui_base.run(title='example_turtle',
						trans=[0, 0, -30],
						keyboard_callback=keyboard_callback,
						render_callback=render_callback)