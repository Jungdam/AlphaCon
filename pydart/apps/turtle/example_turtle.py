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
import profile
import multiprocessing as mp
import time

import threading
import Queue

np.set_printoptions(precision=3)
flag = {}
flag['Train'] = False
cnt_target_update = 0
max_target_update = 20
log_dir = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/tensorflow/log'
ckpt_dir = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/tensorflow/model/'
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle.skel'
warmup_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/warmup/0.1_5000_10.warmup'

pydart.init()
print('pydart initialization OK')

profile = profile.Profile()
threadLock = threading.Lock()

def gen_warmup_data_one_process(q, idx, envi, mu, sigma, num_episode, num_wingbeat):
	data = []
	for ep in xrange(num_episode):
		envi.set_random_state()
		for i in xrange(num_wingbeat):
			action = ac.random(mu,sigma)
			q_skel_init = envi.skel.q
			envi.skel.controller.add_action(action)
			envi.step_forward_wingbeat()
			q_skel_term = envi.skel.q
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
		envi = Envi(dt, skel_file)
		p = mp.Process(
			target=gen_warmup_data_one_process, 
			args=(q, i, envi, mu, sigma, num_episode/num_cores, num_wingbeat))
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

class Target:
	def __init__(self, mu=np.array([0,1.0,6.0]), sigma=np.array([1.5,1.5,2.0])):
		self.mu = mu
		self.sigma = sigma
		self.pos = np.random.normal(self.mu,self.sigma)
	def reset(self):
		self.pos = np.random.normal(self.mu,self.sigma)
	def get_pos(self):
		return self.pos

class Envi:
	def __init__(self, dt, skel_file, num_init_wingbeat=2):
		self.world = pydart.create_world(dt, skel_file)
		self.skel = self.world.skels[0]
		self.skel.controller = controller.Controller(self.world, self.skel)
		self.target = Target()
		while True:
		    if self.skel.controller.get_num_wingbeat() >= num_init_wingbeat:
		        self.world.push()
		        self.world.push()
		        self.set_random_state()
		        break;
		    self.world.step()
	def set_random_state(self):
		self.reset()
	def reset(self):
		self.world.reset()
		self.skel.controller.reset()
		self.target.reset()
	def goodness(self):
		R,p = mmMath.T2Rp(self.skel.body('trunk').T)
		diff = self.target.get_pos()-p
		l = np.linalg.norm(diff)
		return math.exp(-0.2*l*l)
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
		# return np.hstack([state_skel,state_sensor])
		return p_trunk
	def step_forward(self, apply_controller=True, apply_aero=True):
		self.world.step(apply_controller,apply_aero)
		return self.goodness()
	def step_forward_time(self, delta=1.0/10.0):
		elpased = 0.0
		while elpased < delta:
			self.world.step(True,True)
			elpased += dt
		return self.goodness()
	def step_forward_wingbeat(self, wingbeat=None):
		if wingbeat is not None:
			self.skel.controller.add_action(wingbeat,True)
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

def step_forward_wingbeat_unit(q, envi, wingbeat):
	# proc_name = mp.current_process().name
	# print proc_name, 'start', envi.skel.body('trunk').world_com()
	result = envi.step_forward_wingbeat(wingbeat)
	q.put(result)
	# print proc_name, 'end', envi.skel.body('trunk').world_com()

class Envi_Client(mp.Process):
	def __init__(self, q_signal, q_wingbeat, q_result, num_init_wingbeat):
		super(Envi_Client, self).__init__()
		self.envi = Envi(dt, skel_file, num_init_wingbeat)
		self.response_time = 0.001
		self.reward = 0.0
		self.q_signal = q_signal
		self.q_wingbeat = q_wingbeat
		self.q_result = q_result
		self.proc_name = mp.current_process().name
	def error(self, msg):
		print self.proc_name, msg
	def run(self):
		while True:
			if self.q_signal.empty():
				time.sleep(self.response_time)
				continue
			signal = self.q_signal.get()
			if signal == "step":
				if self.q_wingbeat.empty():
					self.error("no wingbeat - "+signal)
					continue
				state_init = self.envi.state()
				wingbeat = self.q_wingbeat.get()
				self.reward = self.envi.step_forward_wingbeat(wingbeat)
				state_term = self.envi.state()
				self.q_result.put([state_init, wingbeat, [self.reward], state_term])
			elif signal == "reward":
				self.q_result.put(self.reward)
			elif signal == "state":
				self.q_result.put(self.envi.state())
			elif signal == "reset":
				self.envi.reset()
			elif signal == "terminate":
				return
			else:
				self.error("unknown signal - "+signal)

class Envi_Server:
	def __init__(self, num_client, dt, skel_file, num_init_wingbeat=2):
		self.num_client = num_client
		self.q_signals = []
		self.q_inputs = []
		self.q_results = []
		self.clients = []
		for i in range(num_client):
			q_signal = mp.Queue()
			q_input = mp.Queue()
			q_result = mp.Queue()
			client = Envi_Client(q_signal, q_input, q_result, num_init_wingbeat)
			self.q_signals.append(q_signal)
			self.q_inputs.append(q_input)
			self.q_results.append(q_result)
			self.clients.append(client)
			client.start()
	def __del__(self):
		for c in self.clients:
			c.terminate()
	def check_empty_result(self):
		for q in self.q_results:
			if not q.empty():
				print "[Envi_Server] error - q is not empty"
				return False
		return True
	def run(self, signals, inputs=None):
		result = []
		if not self.check_empty_result():
			return result
		for i in range(self.num_client):
			if inputs is not None:
				self.q_inputs[i].put(inputs[i])
			self.q_signals[i].put(signals[i])
		for q in self.q_results:
			while q.empty():
				continue
			result.append(q.get())
		return result
	def state(self):
		return self.run(self.num_client*["state"])
	def reward(self):
		return self.run(self.num_client*["reward"])
	def step_forward_wingbeat(self, wingbeats):
		return self.run(self.num_client*["step"], wingbeats)
	def reset(self, idx=None):
		if idx is not None:
			self.q_signals[i].put("reset")
		else:
			for q in self.q_signals:
				q.put("reset")

class Envi_Multicore:
	def __init__(self, num_envis, dt, skel_file, num_init_wingbeat=2):
		self.envis = []
		self.num_envis = num_envis
		for i in range(num_envis):
			self.envis.append(Envi(dt, skel_file, num_init_wingbeat))
		self.num_cores = multiprocessing.cpu_count()
	def set_random_state(self):
		for envi in self.envis:
			envi.set_random_state()
	def goodness(self):
		data = []
		for envi in self.envis:
			data.append(envi.goodness())
		return data
	def state(self):
		return [envi.state() for envi in self.envis]
	def step_forward_wingbeat(self, wingbeats, fast=True):
		rewards = []
		if fast:
			ps = []
			q = mp.Queue()
			for i in xrange(len(wingbeats)):
				p = mp.Process(
					target=step_forward_wingbeat_unit, 
					args=(q, self.envis[i], wingbeats[i]))
				p.start()
				ps.append(p)
			# for i in xrange(len(ps)):
			#  	ps[i].join()
			for i in xrange(len(ps)):
				rewards.append(q.get())
			# for i in xrange(len(ps)):
			#  	ps[i].terminate()

			# threads=[]
			# q = Queue.Queue()
			# for i in xrange(len(wingbeats)):
			# 	t = thread.start_new_thread(
			# 		step_forward_wingbeat_unit,
			# 		(q, self.envis[i], wingbeats[i]))
			# 	threads.append(t)
			# # for t in threads:
			# # 	t.join()
		else:
			for i in xrange(len(wingbeats)):
				r = self.envis[i].step_forward_wingbeat(wingbeats[i])
				rewards.append(r)
		return rewards
	def render(self):
		for envi in self.envis:
			envi.render()
	def get(self, idx):
		return self.envis[idx]
	def reset(self, idx):
		self.envis[idx].set_random_state()

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

class DeepRL_Multicore(deepRL.DeepRLBase):
	
	class Run_Thread(threading.Thread):
		def __init__(self, lock, q, envi, rl, max_iter):
			threading.Thread.__init__(self)
			self.lock = lock
			self.q = q
			self.envi = envi
			self.rl = rl
			self.max_iter = max_iter
		def run(self):
			self.envi.set_random_state()
			for i in range(self.max_iter):
				buffer_name, datum = self.step()
				if datum is None:
					self.envi.set_random_state()
				else:
					self.lock.acquire()
					self.q.put([buffer_name, datum])
					self.lock.release()			
		def step(self):
			# print 'a0'
			state_init = self.envi.state()
			# print 'a1'
			self.lock.acquire()
			# print 'a2'
			action = self.rl.get_action(state_init)
			# print 'a3'
			self.lock.release()
			# print 'a4'
			buffer_name = 'critic'
			is_exploration = self.rl.determine_exploration()
			if is_exploration:
				mu = np.zeros(ac.length())
				sigma = 0.05*np.ones(ac.length())
				action += np.random.normal(mu, sigma)
				buffer_name = 'actor'
			# print 'a5'
			reward = self.envi.step_forward_wingbeat(action)
			# print 'a6'
			state_term = self.envi.state()
			# print 'a7'

			t = [state_init, action, [reward], state_term]

			if basics.check_valid_data(t):
				return buffer_name, t
			else:
				return buffer_name, None
	def __init__(self, envi, nn, warmup_file=None):
		deepRL.DeepRLBase.__init__(self, warmup_file)
		self.replay_buffer['actor'] = deepRL.ReplayBuffer(1000000)
		self.replay_buffer['critic'] = deepRL.ReplayBuffer(1000000)
		self.envi = envi
		self.nn = nn
		self.warmup_size = 0
		self.max_data_gen = 2000000
		self.sample_size = 32
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
			# raw_data = gen_warmup_data(ac.default, [0.1]*ac.dim(), 4000, 20)
			# data = self.convert_warmup_file_to_buffer_data(self.warmup_file)
			# self.replay_buffer['actor'].append(data,verbose=True)
			# self.warmup_size = self.replay_buffer['actor'].size_accum
		else:
			print '[DeepRL]', 'loading warmup file ...'
			data = self.convert_warmup_file_to_buffer_data(self.envi.get(0), self.warmup_file)
			self.replay_buffer['actor'].append(data,verbose=True)
			self.warmup_size = self.replay_buffer['actor'].size_accum
		# self.train_action(self.warmup_size/10, False, 10)
		self.save_variables()
	def convert_warmup_file_to_buffer_data(self, envi, file_name):
		f = open(file_name, 'r')
		data = pickle.load(f)
		size = len(data)
		tuples = []
		cnt = 0
		for d in data:
			q_skel_init = d[0]
			q_skel_term = d[1]
			action = d[2]

			envi.set_random_state()

			envi.world.skel.set_positions(q_skel_init)
			reward_init = envi.step_forward(False,False)
			state_init = envi.state()

			envi.world.skel.set_positions(q_skel_term)
			reward_term = envi.step_forward(False,False)
			state_term = envi.state()

			reward = reward_term
			action_delta = np.array(basics.flatten(action)) - ac.default

			t = [state_init, action_delta, [reward], state_term]
			if basics.check_valid_data(t):
				tuples.append(t)
				cnt += 1
				if cnt%5000 == 0:
					print cnt, ' data were loaded'
		return tuples
	# def run_atomic(self, q, idx, envi, num_ep, num_iter):
	# 	proc_name = multiprocessing.current_process().name
	# 	print proc_name
	# 	data = []
	# 	for i in xrange(num_ep):
	# 		self.init_step(envi)
	# 		print proc_name, 'a0'
	# 		buffer_name, datum = self.step(envi, self.is_warming_up())
	# 		print proc_name, 'a1'
	# 		if datum is None:
	# 			self.init_step()
	# 		else:
	# 			data.append([buffer_name, datum])
	# 	q.put(data)
	def run(self, max_iter=32, verbose=True):
		if self.is_finished_trainning():
			return
		
		# self.init_step()
		# for i in xrange(max_iter):
		# 	buffer_names, data = self.step(self.is_warming_up())
		# 	for j in xrange(len(data)):
		# 		if data[j] is None:
		# 			self.envi.reset(j)
		# 		else:
		# 			self.replay_buffer[buffer_names[j]].append([data[j]])
		# 	if not self.is_warming_up():
		# 		self.train()
		# 	# Print statistics
		# 	if verbose:
		# 		print '[ ', i, 'th episode ]', ' warmup: ', self.is_warming_up(), 
		# 		for buffer_name in self.replay_buffer.keys():
		# 			print '[', buffer_name,
		# 			print self.replay_buffer[buffer_name].size,
		# 			print self.replay_buffer[buffer_name].size_accum, ']',
		# 		if not self.is_warming_up():
		# 			self.print_loss()
		# 		else:
		# 			print ' '

		threads=[]
		q = Queue.Queue()
		for i in range(self.envi.num_envis):
			t = DeepRL_Multicore.Run_Thread(threadLock, q, self.envi.get(i), self, max_iter)
			t.start()
			threads.append(t)
		for t in threads:
			t.join()
		while not q.empty():
			buffer_name, datum = q.get()
			self.replay_buffer[buffer_name].append([datum])

		if not self.is_warming_up():
			self.train()

		if verbose:
			print '[ episode ]', ' warmup: ', self.is_warming_up(), 
			for buffer_name in self.replay_buffer.keys():
				print '[', buffer_name,
				print self.replay_buffer[buffer_name].size,
				print self.replay_buffer[buffer_name].size_accum, ']',
			if not self.is_warming_up():
				self.print_loss()
			else:
				print ' '
	def init_step(self):
		self.envi.set_random_state()
	def step(self, full_random=False):
		# proc_name = multiprocessing.current_process().name
		# print 'get_action0: %s' % (proc_name)
		
		state_inits = self.envi.state()
		actions = self.get_actions(state_inits)

		# print actions

		buffer_names = []
		for i in range(len(actions)):
			buffer_name = 'critic'
			is_exploration = self.determine_exploration()
			if full_random:
				mu = np.zeros(ac.length())
				sigma = 0.1*np.ones(ac.length())
				actions[i] = np.random.normal(mu, sigma)
				buffer_name = 'actor'
			elif is_exploration:
				mu = np.zeros(ac.length())
				sigma = 0.05*np.ones(ac.length())
				actions[i] += np.random.normal(mu, sigma)
				buffer_name = 'actor'
			buffer_names.append(buffer_name)		
		rewards = self.envi.step_forward_wingbeat(actions)
		state_terms = self.envi.state()
		
		tuples = []
		for i in range(len(state_inits)):
			t = [state_inits[i], actions[i], [rewards[i]], state_terms[i]]
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
					if target_qvalue[i][0] > qvalue[i][0]:
						train_state.append(data_state[i])
						train_action.append(data_action[i])
				data_state = train_state
				data_action = train_action
			if len(data_state)>0:
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
	def get_actions(self, states):
		val = self.nn.eval_action([states], True)
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
		print 'Loss values: ', 'qvalue:', q, 'action:', a
	def save_variables(self):
		self.nn.save_variables()

class DeepRL(deepRL.DeepRLBase):
	def __init__(self, envi, nn, warmup_file=None):
		deepRL.DeepRLBase.__init__(self, warmup_file)
		self.replay_buffer['actor'] = deepRL.ReplayBuffer(1000000)
		self.replay_buffer['critic'] = deepRL.ReplayBuffer(1000000)
		self.envi = envi
		self.nn = nn
		self.warmup_size = 0
		self.max_data_gen = 2000000
		self.sample_size = 32
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
			# raw_data = gen_warmup_data(ac.default, [0.1]*ac.dim(), 4000, 20)
			# data = self.convert_warmup_file_to_buffer_data(self.warmup_file)
			# self.replay_buffer['actor'].append(data,verbose=True)
			# self.warmup_size = self.replay_buffer['actor'].size_accum
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
			action_delta = np.array(basics.flatten(action)) - ac.default

			t = [state_init, action_delta, [reward], state_term]
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
		if full_random:
			mu = np.zeros(ac.length())
			sigma = 0.1*np.ones(ac.length())
			action = np.random.normal(mu, sigma)
			buffer_name = 'actor'
		elif is_exploration:
			mu = np.zeros(ac.length())
			sigma = 0.05*np.ones(ac.length())
			action += np.random.normal(mu, sigma)
			buffer_name = 'actor'
		reward = self.envi.step_forward_wingbeat(action)
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
myNN.initialize([len(myEnvi.state()),len(ac.default)])


myEnviServer = Envi_Server(2, dt, skel_file)
print myEnviServer.state()
print myEnviServer.reward()

for i in range(5):
	a = []
	for i in range(2):
		a.append(ac.default)
	r = myEnviServer.step_forward_wingbeat(a)
	print r

time.sleep(1000)

# # myNN.initialize([len(myEnvi.state()),2],ckpt_dir)
# myDeepRL = DeepRL(myEnvi, myNN, warmup_file)
# envis = []
# for i in range(multiprocessing.cpu_count()):
# 	envis.append(Envi(dt, skel_file))
num_envis = 32
num_episode = 16
myEnviMulti = Envi_Multicore(num_envis, dt, skel_file)
# myDeepRL = DeepRL_Multicore(myEnviMulti, myNN, warmup_file)
myDeepRL = DeepRL_Multicore(myEnviMulti, myNN)

def step_callback():
	return

def render_callback():
	gl_render.render_ground(color=[0.3,0.3,0.3],axis='x')
	gl_render.render_ground(color=[0.3,0.3,0.3],axis='y')
	gl_render.render_ground(color=[0.3,0.3,0.3],axis='z')
	myEnvi.render()
	# myEnviMulti.render()
	# gl_render.render_ground(color=[1.0,1.0,1.0],axis='z')
	# myEnvi.detective.render(color=[0,0,1])
	# myEnvi.criminal.render(color=[1,0,0])
	if flag['Train']:
		global cnt_target_update
		myDeepRL.run(num_episode)
		if cnt_target_update>=max_target_update:
			myDeepRL.save_variables()
			cnt_target_update = 0
			print '------Target Network is updated------'
		else:
			cnt_target_update += 1

def keyboard_callback(key):
	if key == 'r':
		myEnvi.set_random_state()
		myEnviMulti.set_random_state()
	elif key == 'p':
		elpased = 0.0
		while True:
			if myEnvi.skel.controller.is_new_wingbeat():
				print myEnvi.skel.controller.action[-1]
			myEnvi.step_forward()
			elpased += dt
			if elpased >= 0.1:
				break
	# elif key == '[':
		# elpased = 0.0
		# while True:
		# 	if myEnvi.skel.controller.is_new_wingbeat():
		# 		myEnvi.skel.controller.add_action(ac.random(ac.default,[0.1]*ac.dim()))
		# 	myEnvi.step_forward()
		# 	elpased += dt
		# 	if elpased >= 0.1:
		# 		break
	# elif key == ']':
	# 	wingbeat=[]
	# 	for i in range(num_envis):
	# 		wingbeat.append(ac.random(ac.default,[0.1]*ac.dim()))
	# 	myEnviMulti.step_forward_wingbeat(wingbeat, False)
	elif key == 't':
		print myEnvi.state()
	elif key == ' ':
		elpased = 0.0
		while True:
			if myEnvi.skel.controller.is_new_wingbeat():
				state = myEnvi.state()
				action = myDeepRL.get_action(state)
				qvalue = myDeepRL.get_qvalue(state)
				reward = myEnvi.goodness()
				print 'S:', state, 'A:', action, 'R:', reward, 'Q:', qvalue
				myEnvi.skel.controller.add_action(action, True)
			myEnvi.step_forward()
			elpased += dt
			if elpased >= 0.1:
				break
	elif key == 'd':
		flag['Train'] = not flag['Train']
		print 'Train: ', flag['Train']
	elif key == 's':
		myNN.save(ckpt_dir)
	elif key == 'w':
		gen_warmup_data(ac.default, [0.1]*ac.dim(), 5000, 10)
	elif key == '0':
		sample_idx = myDeepRL.replay_buffer['critic'].sample_idx(10)
		data = myDeepRL.sample('critic', sample_idx)
		print data

pydart.glutgui.glutgui_base.run(title='example_turtle',
						trans=[0, 0, -30],
						keyboard_callback=keyboard_callback,
						render_callback=render_callback)