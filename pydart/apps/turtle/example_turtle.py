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
import environment as env
import tensorflow as tf
import mmMath
import action as ac
import pickle
import profile
import multiprocessing as mp
import time

np.set_printoptions(precision=3)
flag = {}
flag['Train'] = False
cnt_target_update = 0
max_target_update = 20
log_dir = './data/tensorflow/log'
ckpt_load_dir = None
ckpt_load_dir = './data/tensorflow/model/'
ckpt_save_dir = './data/tensorflow/model/'
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle.skel'
warmup_file = None
warmup_file = './data/warmup/0.6_5000_10_noprior.warmup'

num_init_wingbeat = 0
dt = 1.0/1000.0
max_client = 16
max_steps = 20

pydart.init()
print('pydart initialization OK')

profile = profile.Profile()

# class Target:
# 	def __init__(self, base=np.array([0.0,1.0,3.0]), dist=6.0, span=0.3333*math.pi):
# 		self.base = base
# 		self.dist = dist
# 		self.span = span
# 		self.pos = self.new_pos()
# 	def reset(self):
# 		self.pos = self.new_pos()
# 	def new_pos(self):
# 		z = np.random.uniform(0,self.dist)
# 		r = z*math.tan(0.5*self.span)
# 		x = np.random.uniform(-r,r)
# 		y = np.random.uniform(-r,r)
# 		return self.base+np.array([x,y,z])
# 	def get_pos(self):
# 		return self.pos

class Target:
	def __init__(self, 
		base=np.array([0.0,1.0,0.0]), 
		offset=np.array([0.0,0.0,0.0]),
		sigma=np.array([2.0,2.0,3.0])):
		self.base = base
		self.offset = offset
		self.sigma = sigma
		self.pos = self.new_pos()
	def reset(self):
		self.pos = self.new_pos()
	def new_pos(self):
		return self.base+np.random.normal(self.offset, self.sigma)
	def get_pos(self):
		return self.pos
	def set_pos(self, pos):
		self.pos = pos

class Env(env.EnvironmentBase):
	def __init__(self, dt, skel_file, num_init_wingbeat=2):
		self.world = pydart.create_world(dt, skel_file)
		self.skel = self.world.skels[0]
		self.skel.controller = controller.Controller(self.world, self.skel)
		self.target = Target()
		while True:
		    if self.skel.controller.get_num_wingbeat() >= num_init_wingbeat:
		        self.world.push()
		        self.world.push()
		        self.reset()
		        break;
		    self.world.step()
	def reset(self):
		self.world.reset()
		self.skel.controller.reset()
		self.target.reset()
	def goodness(self):
		R,p = mmMath.T2Rp(self.skel.body('trunk').T)
		diff = self.target.get_pos()-p
		l = np.linalg.norm(diff)
		return math.exp(-1.0*l*l)
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
		state_sensor = np.dot(R_trunk_inv,self.target.get_pos()-p_trunk)
		# return state_sensor
		return np.hstack([state_skel,state_sensor])
	def step_forward(self, apply_controller=True, apply_aero=True):
		self.world.step(apply_controller,apply_aero)
		return self.goodness()
	def step(self, wingbeat=None):
		if wingbeat is not None:
			self.skel.controller.add_action(wingbeat)
		while True:
			self.world.step(True,True)
			if self.skel.controller.is_new_wingbeat():
				break
		# print proc_name, 'end'
		return self.goodness()
	def render(self):
		if self.world is not None:
			glEnable(GL_LIGHTING)
			# glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
			self.world.render()
			glDisable(GL_LIGHTING)
		glColor3d(1.0, 0.0, 0.0)
		p = self.target.get_pos()
		glPushMatrix()
		glTranslated(p[0],p[1],p[2])
		glutSolidSphere(0.25, 10, 10)
		glPopMatrix()

class Env_Slave_Custom(env.Environment_Slave):
	def __init__(self, idx, q_input, q_result, func_gen_env, args_gen_env):
		super(Env_Slave_Custom, self).__init__(
			idx, q_input, q_result, func_gen_env, args_gen_env)
	def run_extra(self, signal, data):
		if signal == "set_target_pos":
			self.env.target.set_pos(data)
		elif signal == "get_target_pos":
			p = self.env.target.get_pos()
			self.q_result.put(p)

class En_Master_Custom(env.Environment_Master):
	def get_target_pos(self):
		return self.run(self.num_slave*["get_target_pos"])
	def set_target_pos(self, pos):
		for i in range(self.num_slave):
			self.q_inputs[i].put(["set_target_pos",pos[i]])

def gen_env(args):
	dt = args[0]
	skel_file = args[1]
	if len(args)>2:
		num_init_wingbeat = args[2]
		return Env(dt, skel_file, num_init_wingbeat)
	else:
		return Env(dt, skel_file)

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
		self.learning_rate = 1.0*1e-2
		self.writer = None
		self.merged = None
	def initialize(self, data, ckpt_file=None):
		tf.reset_default_graph()
		with self.graph.as_default():
			# state dimension
			d = data[0]
			# action dimension
			a = data[1]
			# placeholders for inputs
			state = tf.placeholder(tf.float32, [None,d])
			target_qvalue = tf.placeholder(tf.float32, [None,1])
			target_action = tf.placeholder(tf.float32, [None,a])
			keep_prob = tf.placeholder(tf.float32)
			
			# Network definition
			layer_1 = nn.Layer('layer_1',self.var, False, state, d, 64)
			layer_2 = nn.Layer('layer_2',self.var, False, layer_1.h, 64, 128)
			layer_3 = nn.Layer('layer_3',self.var, False, layer_2.h, 128, 32, 
				dropout_enabled=True, dropout_placeholder=keep_prob)
			layer_q = nn.Layer('layer_q',self.var, False, layer_3.h, 32, 1, None)
			layer_a = nn.Layer('layer_a',self.var, False, layer_3.h, 32, a, None)

			layer_1_copy = layer_1.copy(state)
			layer_2_copy = layer_2.copy(layer_1_copy.h)
			layer_3_copy = layer_3.copy(layer_2_copy.h)
			layer_q_copy = layer_q.copy(layer_3_copy.h)
			layer_a_copy = layer_a.copy(layer_3_copy.h)

			h_qvalue = layer_q.h
			h_action = layer_a.h
			h_qvalue_copy = layer_q_copy.h
			h_action_copy = layer_a_copy.h
			
			# Optimizer
			loss_qvalue = tf.reduce_mean(tf.square(target_qvalue - h_qvalue))
			loss_action = tf.reduce_mean(tf.square(target_action - h_action))			
			
			# Exponential decay
			# global_step = tf.Variable(0, trainable=False)
			# learning_rate = tf.train.exponential_decay(
			# 	self.learning_rate, global_step, 1000000, 0.96, staircase=True)
			
			# Optimization for trainning
			opt_q = tf.train.GradientDescentOptimizer(self.learning_rate)
			opt_a = tf.train.GradientDescentOptimizer(self.learning_rate)
			# opt_q = tf.train.AdamOptimizer(self.learning_rate)
			# opt_a = tf.train.AdamOptimizer(self.learning_rate)
			grads_and_vars_q = opt_q.compute_gradients(loss_qvalue)
			grads_and_vars_a = opt_a.compute_gradients(loss_action)
			
			# Gradient clipping
			capped_grads_and_vars_q = [\
				(tf.clip_by_norm(grad,50.0), var) if grad is not None \
				else (None, var) for grad, var in grads_and_vars_q]
			capped_grads_and_vars_a = [\
				(tf.clip_by_norm(grad,50.0), var) if grad is not None \
				else (None, var) for grad, var in grads_and_vars_a]
			
			# Gradient applying
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

class DeepRL_Multicore(deepRL.DeepRLBase):
	def __init__(self, env, nn, warmup_file=None):
		deepRL.DeepRLBase.__init__(self, warmup_file)
		self.replay_buffer['actor'] = deepRL.ReplayBuffer(500000)
		self.replay_buffer['critic'] = deepRL.ReplayBuffer(500000)
		self.env = env
		self.nn = nn
		self.warmup_size = 0
		self.max_data_gen = 1000000
		self.sample_size = 32
		self.target_pos_pool = []
		while len(self.target_pos_pool) < self.env.num_slave:
			self.env.reset()
			self.target_pos_pool += self.env.get_target_pos()
		if warmup_file is None:
			print '[DeepRL]', 'generating warmup data ...'
			# cnt = 0
			# while True:
			# 	self.init_step()
			# 	buffer_name, datum = self.step(full_random=True)
			# 	if datum is None:
			# 		self.init_step()
			# 	else:
			# 		self.replay_buffer['actor'].append([datum])
			# 		cnt += 1
			# 	if cnt >= self.warmup_size:
			# 		break
			# 	if cnt%1000 == 0:
			# 		print cnt, ' data were generated'
			# raw_data = gen_warmup_data(ac.default, [0.1]*ac.dim, 4000, 20)
			# data = self.convert_warmup_file_to_buffer_data(self.warmup_file)
			# self.replay_buffer['actor'].append(data,verbose=True)
			# self.warmup_size = self.replay_buffer['actor'].size_accum
		else:
			print '[DeepRL]', 'loading warmup file ...'
			data = self.convert_warmup_file_to_buffer_data(Env(dt, skel_file), self.warmup_file)
			self.replay_buffer['actor'].append(data,verbose=True)
			self.replay_buffer['critic'].append(data,verbose=True)
			self.warmup_size = self.replay_buffer['actor'].size_accum
		# for i in range(1000):
		# 	self.train_qvalue(self.sample_size)
		# 	self.train_action(self.sample_size)
		# 	if i%20==0:
		# 		self.save_variables()
		# 	if i%100==0:
		# 		self.print_loss()
		# 		print ' '
		self.save_variables()
	def convert_warmup_file_to_buffer_data(self, env, file_name):
		f = open(file_name, 'r')
		data = pickle.load(f)
		size = len(data)
		tuples = []
		cnt = 0
		for d in data:
			q_skel_init = d[0]
			q_skel_term = d[1]
			action = d[2]

			env.reset()

			env.world.skel.set_positions(q_skel_init)
			reward_init = env.step_forward(False,False)
			state_init = env.state()

			env.world.skel.set_positions(q_skel_term)
			reward_term = env.step_forward(False,False)
			state_term = env.state()

			reward = reward_term
			action_delta = action-ac.default

			t = [state_init, action_delta, [reward], state_term]
			if basics.check_valid_data(t):
				tuples.append(t)
				cnt += 1
				if cnt%5000 == 0:
					print cnt, ' data were loaded'
		return tuples
	def run(self, max_steps=32, verbose=True):
		if self.is_finished_trainning():
			return
		self.init_step()
		for i in range(max_steps):
			buffer_names, data = self.step(self.is_warming_up())
			for j in range(len(data)):
				if data[j] is None:
					self.env.reset(j)
				else:
					self.replay_buffer[buffer_names[j]].append([data[j]])
			if not self.is_warming_up():
				self.train()
			# Print statistics
			if verbose:
				print '[ ', i, 'th steps ]', ' warmup: ', self.is_warming_up(), 
				for buffer_name in self.replay_buffer.keys():
					print '[', buffer_name,
					print self.replay_buffer[buffer_name].size,
					print self.replay_buffer[buffer_name].size_accum, ']',
				if not self.is_warming_up():
					self.print_loss()
				print ' '
	def measure_controller_quality(self, max_steps):
		num_slave = self.env.num_slave
		num_iter = len(self.target_pos_pool)/num_slave
		cnt_wingbeat = 0
		sum_reward = 0.0
		for i in range(num_iter):
			self.env.set_target_pos(self.target_pos_pool[num_slave*i:num_slave*(i+1)])
			for j in range(max_steps):
				buffer_names, data = self.step(force_critic=True)
				for k in range(len(data)):
					if data[k] is None:
						continue
					cnt_wingbeat += 1
					sum_reward += data[k][2][0]
		return sum_reward/cnt_wingbeat
	def init_step(self):
		self.env.reset()
	def step(self, full_random=False, force_critic=False):
		state_inits = self.env.state()
		actions = self.get_actions(state_inits, ac.default)
		buffer_names = []
		for i in range(self.env.num_slave):
			buffer_name = 'critic'
			if not force_critic:
				is_exploration = self.determine_exploration()
				if full_random:
					actions[i] = ac.random(ac.default,
						self.get_random_noise()*np.ones(ac.dim))
					buffer_name = 'actor'
				elif is_exploration:
					actions[i] += np.random.normal(np.zeros(ac.dim),
						[self.get_exploration_noise()]*ac.dim)
					actions[i] = ac.clamp(actions[i])
					buffer_name = 'actor'
			buffer_names.append(buffer_name)
		rewards = self.env.step(actions)
		state_terms = self.env.state()
		tuples = []
		for i in range(self.env.num_slave):
			act = actions[i]-ac.default
			t = [state_inits[i], act, [rewards[i]], state_terms[i]]
			if basics.check_valid_data(t):
				tuples.append(t)
			else:
				tuples.append(None)
		return buffer_names, tuples
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
					if target_qvalue[i][0] > qvalue[i][0] + self.get_qvalue_knoll():
						print target_qvalue[i][0], qvalue[i][0]
						train_state.append(data_state[i])
						train_action.append(data_action[i])
				data_state = train_state
				data_action = train_action
			if len(data_state)>0:
				print 'Train Action Size:', len(data_state)
				for i in range(iteration):
					self.nn.train_action([data_state,data_action])
					if verbose:
						print self.nn.loss_action([data_state,data_action])
	def train(self):
		self.train_qvalue(self.sample_size)
		self.train_action(self.sample_size)
	def get_action(self, state, default_action=None):
		val = self.nn.eval_action([[state]], True)
		if default_action is not None:
			return val[0] + default_action
		else:
			return val[0]
	def get_actions(self, states, default_action=None):
		val = self.nn.eval_action([states], True)
		if default_action is not None:
			for i in range(len(val)):
				val[i] += default_action
		return val
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
		print 'Loss values: ', 'qvalue:', q, 'action:', a,
	def save_variables(self):
		self.nn.save_variables()


myEnvi = Env(dt, skel_file, num_init_wingbeat)
myNN = NN('net_turtle')
myNN.initialize([len(myEnvi.state()),len(ac.default)], ckpt_load_dir)

myEnviMaster = En_Master_Custom(
	max_client, 
	gen_env, 
	[dt, skel_file, num_init_wingbeat],
	Env_Slave_Custom)

myDeepRL = DeepRL_Multicore(myEnviMaster, myNN, warmup_file)

def gen_warmup_data_one_process(q, idx, env, mu, sigma, num_episode, num_wingbeat):
	data = []
	for ep in xrange(num_episode):
		env.reset()
		for i in xrange(num_wingbeat):
			action = ac.random(mu,sigma)
			q_skel_init = env.skel.q
			env.step(action)
			q_skel_term = env.skel.q
			data.append([q_skel_init, q_skel_term, action])
			if len(data)%100 == 0:
				print '[', idx, ']', len(data), 'data generated'
	q.put(data)

def gen_warmup_data(mu, sigma, num_episode=2500, num_wingbeat=20):
	num_cores = mp.cpu_count()
	data = []
	ps = []
	q = mp.Queue()
	for i in xrange(num_cores):
		env = Env(dt, skel_file, num_init_wingbeat)
		p = mp.Process(
			target=gen_warmup_data_one_process, 
			args=(q, i, env, mu, sigma, num_episode/num_cores, num_wingbeat))
		p.start()
		ps.append(p)
	for i in xrange(num_cores):
		data = data + q.get()
	for i in xrange(num_cores):
		ps[i].join()
	for i in xrange(num_cores):
		ps[i].terminate()

	f = open(str(sigma[0])+'_'+str(num_episode)+'_'+str(num_wingbeat)+'.warmup', 'w')
	pickle.dump(data, f)
	f.close()

	return data

def step_callback():
	return

def render_callback():
	gl_render.render_ground(color=[0.3,0.3,0.3],axis='x')
	gl_render.render_ground(color=[0.3,0.3,0.3],axis='y')
	gl_render.render_ground(color=[0.3,0.3,0.3],axis='z')
	myEnvi.render()
	if flag['Train']:
		global cnt_target_update
		myDeepRL.run(max_steps)
		if not myDeepRL.is_finished_trainning():
			if cnt_target_update>=max_target_update:
				myDeepRL.save_variables()
				cnt_target_update = 0
				print '------Target Network is updated------',
				print 'avg_reward:', myDeepRL.measure_controller_quality(max_steps)
			else:
				cnt_target_update += 1

def keyboard_callback(key):
	if key == 'r':
		myEnvi.reset()
	elif key == 't':
		print myEnvi.state()
	elif key == 'p':
		elpased = 0.0
		while True:
			if myEnvi.skel.controller.is_new_wingbeat():
				print myEnvi.skel.controller.action[-1]
			myEnvi.step_forward()
			elpased += dt
			if elpased >= 0.1:
				break
	elif key == '[':
		elpased = 0.0
		while True:
			if myEnvi.skel.controller.is_new_wingbeat():
				state = myEnvi.state()
				reward = myEnvi.goodness()
				action = ac.random(ac.default,
					[myDeepRL.get_random_noise()]*ac.dim)
				myEnvi.skel.controller.add_action(action)
				print 'S:', state, 'A:', action, 'R:', reward
			myEnvi.step_forward()
			elpased += dt
			if elpased >= 0.1:
				break
	elif key == ' ':
		elpased = 0.0
		while True:
			if myEnvi.skel.controller.is_new_wingbeat():
				state = myEnvi.state()
				action = myDeepRL.get_action(state, ac.default)
				qvalue = myDeepRL.get_qvalue(state)
				reward = myEnvi.goodness()
				myEnvi.skel.controller.add_action(action)
				print 'S:', state, 'A:', action, 'R:', reward, 'Q:', qvalue
			myEnvi.step_forward()
			elpased += dt
			if elpased >= 0.1:
				break
	elif key == 'd':
		flag['Train'] = not flag['Train']
		print 'Train: ', flag['Train']
	elif key == 's':
		myNN.save(ckpt_save_dir)
	elif key == 'w':
		gen_warmup_data(ac.default, 
			[myDeepRL.get_random_noise()]*ac.dim, 5000, 10)
	elif key == '0':
		sample_idx = myDeepRL.replay_buffer['critic'].sample_idx(10)
		data = myDeepRL.sample('critic', sample_idx)
		print data

pydart.glutgui.glutgui_base.run(title='example_turtle',
						trans=[0, 0, -30],
						keyboard_callback=keyboard_callback,
						render_callback=render_callback)