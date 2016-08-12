from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pydart
import gl_render
import numpy as np
import math
import basics
import deepRL
import nn
import tensorflow as tf

np.set_printoptions(precision=3)
flag = {}
flag['Train'] = False

def R(theta):
	c = math.cos(theta)
	s = math.sin(theta)
	return np.array([[c,-s],[s,c]])

def vel_random(mu=np.array([0.5,0.0]),sigma=np.array([0.25,0.1*math.pi])):
	return np.random.normal(mu, sigma)

class Actor:
	def __init__(self, pos=np.array([0,0]), ori=np.array([1,0])):
		self.pos = pos
		self.ori = ori
	def step_forward(self, dt, vel):
		self.pos = self.pos + vel[0]*self.ori
		self.rotate(vel[1]*dt)
	def set_random_state(self, sigma=[5,5,0.5*math.pi]):
		r = np.random.normal([0,0,0], sigma)
		self.pos = r[0:2]
		self.rotate(r[2])
	def rotate(self, theta):
		self.ori = np.dot(R(theta), self.ori)	
	def render(self, color=[0.5,0.5,0.5]):
		glColor3d(color[0],color[1],color[2])
		glLineWidth(5.0)

		glPushMatrix()
		glTranslated(self.pos[0],self.pos[1],0)
		glutSolidSphere(0.5, 10, 10)
		glBegin(GL_LINES)
		glVertex3d(0,0,0)
		glVertex3d(self.ori[0],self.ori[1],0)
		glEnd()
		glPopMatrix()

class Envi:
	def __init__(self):
		self.detective = Actor()
		self.criminal = Actor()
		self.vel_limit = [np.array([0.0,-0.2*math.pi]),np.array([1.0,0.2*math.pi])]
		self.pos_limit = [np.array([-10.0,-10.0]),np.array([10.0,10.0])]
		self.set_random_state()
	def set_random_state(self, apply_detective=True, apply_criminal=True):
		if apply_detective:
			self.detective.set_random_state()
			self.detective.pos = self.limit_position(self.detective.pos)
		if apply_criminal:
			self.criminal.set_random_state()
			self.criminal.pos = self.limit_position(self.criminal.pos)
	def goodness(self):
		if self.check_collision(self.detective.pos):
			return -1.0
		else:
			diff = self.criminal.pos-self.detective.pos
			l = np.linalg.norm(diff)
			return math.exp(-0.5*l*l)# + 0.1*math.exp(-5.0*((np.dot(self.detective.ori,diff/l)-1.0)**2))
	def state(self):
		diff = self.criminal.pos-self.detective.pos
		x = self.detective.ori[:]
		y = np.dot(R(0.5*math.pi),x)
		R_inv = np.array([x,y])
		criminal_dir = np.dot(R_inv,diff)
		return np.hstack([self.detective.pos, self.detective.ori, criminal_dir])
		# w_dir0 = np.dot(R_inv,np.array([self.pos_limit[0][0],self.pos_limit[0][1]]))
		# w_dir1 = np.dot(R_inv,np.array([self.pos_limit[1][0],self.pos_limit[1][1]]))
		# w_dir2 = np.dot(R_inv,np.array([self.pos_limit[0][0],self.pos_limit[1][1]]))
		# w_dir3 = np.dot(R_inv,np.array([self.pos_limit[1][0],self.pos_limit[0][1]]))
		# return np.hstack([criminal_dir,w_dir0,w_dir1,w_dir2,w_dir3])
	def limit_velocity(self, vel):
		v = max(vel[0], self.vel_limit[0][0])
		v = min(v, self.vel_limit[1][0])
		w = max(vel[1], self.vel_limit[0][1])
		w = min(w, self.vel_limit[1][1])
		return np.array([v,w])
	def check_collision(self, pos):
		return pos[0]<=self.pos_limit[0][0] or \
			pos[0]>=self.pos_limit[1][0] or \
			pos[1]<=self.pos_limit[0][1] or \
			pos[1]>=self.pos_limit[1][1]
	def limit_position(self, pos):
		p_x = max(pos[0], self.pos_limit[0][0])
		p_x = min(p_x, self.pos_limit[1][0])
		p_y = max(pos[1], self.pos_limit[0][1])
		p_y = min(p_y, self.pos_limit[1][1])
		return np.array([p_x,p_y])
	def step_forward(self, dt, vel_detective=None, vel_criminal=None):
		good_init = self.goodness()
		if vel_detective is not None:
			self.detective.step_forward(dt, self.limit_velocity(vel_detective))
			self.detective.pos = self.limit_position(self.detective.pos)
		if vel_criminal is not None:
			self.criminal.step_forward(dt, self.limit_velocity(vel_criminal))
			self.criminal.pos = self.limit_position(self.criminal.pos)
		good_term = self.goodness()
		reward_delta = good_term - good_init
		reward = self.goodness() + reward_delta
		if self.check_collision(self.detective.pos):
			return -1.0
		else:
			return reward

class NNToy(nn.NNBase):
	def __init__(self, name):
		nn.NNBase.__init__(self, name)
		self.dropout_keep_prob = 0.5
		self.train_a = None
		self.train_q = None
		self.eval_q = None
		self.eval_a = None
		self.placeholder_state = None
		self.placeholder_target_qvalue = None
		self.placeholder_target_action = None
		self.placeholder_dropout_keep_prob = None
		self.loss_q = None
		self.loss_a = None
		self.learning_rate = 0.001
		self.var_tar = None
		self.var_cur = None
	def initialize(self, data):
		tf.reset_default_graph()
		with self.graph.as_default():
			d = data[0]		# state dimension
			a = data[1]		# action dimension
			# 
			state = tf.placeholder(tf.float32, [None,d])
			target_qvalue = tf.placeholder(tf.float32, [None,1])
			target_action = tf.placeholder(tf.float32, [None,a])
			keep_prob = tf.placeholder(tf.float32)

			self.var_cur = nn.Variables(self.sess)
			# 1st layer
			W_fc1 = self.var_cur.weight_variable('W_fc1',[d, 32])
			b_fc1 = self.var_cur.bias_variable('b_fc1',[32])
			h_fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
			# 2nd layer
			W_fc2 = self.var_cur.weight_variable('W_fc2',[32, 32])
			b_fc2 = self.var_cur.bias_variable('b_fc2',[32])
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# 3rd layer
			W_fc3 = self.var_cur.weight_variable('W_fc3',[32, 16])
			b_fc3 = self.var_cur.bias_variable('b_fc3',[16])
			h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
			# Layer for Q value
			W_qvalue = self.var_cur.weight_variable('W_qvalue',[16, 1])
			b_qvalue = self.var_cur.bias_variable('b_qvalue',[1])
			h_qvalue = tf.matmul(h_fc3_drop, W_qvalue) + b_qvalue
			# Layer for action
			W_action = self.var_cur.weight_variable('W_action',[16, a])
			b_action = self.var_cur.bias_variable('b_action',[a])
			h_action = tf.matmul(h_fc3_drop, W_action) + b_action

			self.var_tar = nn.Variables(self.sess, self.var_cur)
						

			# # 1st layer
			# W_fc1 = nn.weight_variable('W_fc1',[d, 32])
			# b_fc1 = nn.bias_variable('b_fc1',[32])
			# h_fc1 = tf.nn.relu(tf.matmul(state, W_fc1) + b_fc1)
			# # 2nd layer
			# W_fc2 = nn.weight_variable('W_fc2',[32, 32])
			# b_fc2 = nn.bias_variable('b_fc2',[32])
			# h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# # 3rd layer
			# W_fc3 = nn.weight_variable('W_fc3',[32, 16])
			# b_fc3 = nn.bias_variable('b_fc3',[16])
			# h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			# h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
			# # Layer for Q value
			# W_qvalue = nn.weight_variable('W_qvalue',[16, 1])
			# b_qvalue = nn.bias_variable('b_qvalue',[1])
			# h_qvalue = tf.matmul(h_fc3_drop, W_qvalue) + b_qvalue
			# # Layer for action
			# W_action = nn.weight_variable('W_action',[16, a])
			# b_action = nn.bias_variable('b_action',[a])
			# h_action = tf.matmul(h_fc3_drop, W_action) + b_action

			# Optimizer
			loss_qvalue = tf.reduce_mean(tf.square(target_qvalue - h_qvalue))
			loss_action = tf.reduce_mean(tf.square(target_action - h_action))
			# loss_qvalue = tf.reduce_mean(tf.square(target_qvalue - h_qvalue))
			# loss_action = tf.reduce_mean(tf.square(target_action - h_action))
			# loss_qvalue = tf.reduce_mean(-tf.reduce_sum(target_qvalue * tf.log(h_qvalue), reduction_indices=[1]))
			# loss_action = tf.reduce_mean(-tf.reduce_sum(target_action * tf.log(h_action), reduction_indices=[1]))

			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.train.exponential_decay(
				self.learning_rate, global_step, 1000000, 0.96, staircase=True)

			# self.train_q = tf.train.AdamOptimizer(1e-3).minimize(loss_qvalue)
			# self.train_a = tf.train.AdamOptimizer(1e-3).minimize(loss_action)
			opt_q = tf.train.GradientDescentOptimizer(learning_rate)
			opt_a = tf.train.GradientDescentOptimizer(learning_rate)
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
			# Place holders
			self.placeholder_state = state
			self.placeholder_target_qvalue = target_qvalue
			self.placeholder_target_action = target_action
			self.placeholder_dropout_keep_prob = keep_prob
			# Loss
			self.loss_q = loss_qvalue
			self.loss_a = loss_action
			# Initialize all variables
			self.sess = tf.Session(graph=self.graph)
			self.sess.run(tf.initialize_all_variables())
			self.saver = tf.train.Saver()
			self.initialized = True
	def eval_qvalue(self, data):
		val = self.sess.run(self.eval_q, feed_dict={
			self.placeholder_state: data[0],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def eval_action(self, data):
		val = self.sess.run(self.eval_a, feed_dict={
			self.placeholder_state: data[0],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
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

class DeepRLToy(deepRL.DeepRLBase):
	def __init__(self, envi, nn, warmup_file=None):
		deepRL.DeepRLBase.__init__(self, warmup_file)
		self.replay_buffer['actor'] = deepRL.ReplayBuffer(500000)
		self.replay_buffer['critic'] = deepRL.ReplayBuffer(500000)
		self.envi = envi
		self.nn = nn
		self.warmup_size = 50000
		self.max_data_gen = 10000000
		self.sample_size = 64
		
		cnt = 0
		while True:
			buffer_name, datum = self.step(full_random=True)
			if datum is None:
				self.init_step()
			else:
				self.replay_buffer['actor'].append([datum])
				self.replay_buffer['critic'].append([datum])
				cnt += 1
			if cnt >= self.warmup_size:
				break
			if cnt%1000 == 0:
				print cnt, ' data were generated'
		
		# # Train greedy policy
		# sample_idx = self.replay_buffer['actor'].sample_idx(5000)
		# data = self.sample('actor', sample_idx)
		# s = data[0]
		# a = data[1]
		# r = data[2]
		# t_s = []
		# t_a = []
		# for i in range(len(s)):
		# 	if r[i] > 1.0e-2:
		# 		t_s.append(s[i])
		# 		t_a.append(a[i])
		# print 'Data:', len(t_s)
		# if t_s:
		# 	print 'Init:', self.nn.loss_action([t_s,t_a])
		# 	for i in range(100000):
		# 		self.nn.train_action([t_s,t_a])
		# 		if i%100==0:
		# 			print self.nn.loss_action([t_s,t_a])

		# # Train random policy
		# print '--------- Train random policy ---------'
		# print 'Init Loss:', self.loss_action(buffer_name='actor')
		# num_train_data = 100*self.warmup_size
		# cnt = 0
		# while True:
		# 	for i in range(20):
		# 		self.train_action(100,check_qvalue=False)
		# 	cnt += 1
		# 	if cnt%1 == 0:
		# 		print cnt, ' data were trainned', self.loss_action(buffer_name='actor')
		# 	if cnt>=num_train_data:
		# 		break
		# print '---------------------------------------'
		# sample_idx = self.replay_buffer['actor'].sample_idx(100)
		# data = self.sample('actor', sample_idx)
		# s = data[0]
		# a = data[1]
		# print 'Init Loss:', self.nn.loss_action([s,a]), a[0], self.nn.eval_action([s])[0]
		# for i in range(10000):
		# 	self.nn.train_action([s,a])
		# 	# if i%100==0:
		# 	print i, np.array([self.nn.loss_action([s,a])]), a[0], self.nn.eval_action([s])[0]
	def convert_warmup_file_to_buffer_data(self, file_name):
		return
	def init_step(self):
		self.envi.set_random_state()
	def step(self, full_random=False, force_buffer_name=None):
		buffer_name = 'critic'
		is_exploration = self.determine_exploration()

		state_init = self.envi.state()
		action = self.get_action(state_init)
		if full_random:
			mu = 0.5*(self.envi.vel_limit[0]+self.envi.vel_limit[1])
			sigma = 0.25*(self.envi.vel_limit[1]-self.envi.vel_limit[0])
			action = vel_random(mu,sigma)
			buffer_name = 'actor'
		elif is_exploration:
			mu = np.zeros(len(self.envi.vel_limit[0]))
			sigma = 0.1*(self.envi.vel_limit[1]-self.envi.vel_limit[0])
			action += vel_random(mu,sigma)
			buffer_name = 'actor'
		reward = self.envi.step_forward(dt,action)
		state_term = self.envi.state()
		t = [state_init, action, [reward], state_term]
		# print state_init, action, reward
		if force_buffer_name is not None:
			buffer_name = force_buffer_name
		if basics.check_valid_data(t) and \
			not self.envi.check_collision(self.envi.detective.pos):
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
		qvalue_prime = self.nn.eval_qvalue([state_prime])
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
				qvalue = self.nn.eval_qvalue([data_state])
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
		val = self.nn.eval_action([[state]])
		return val[0]
	def get_qvalue(self, state):
		val = self.nn.eval_qvalue([[state]])
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
dt = 0.5
myEnvi = Envi()
myNN = NNToy('net')
myNN.initialize([len(myEnvi.state()),2])
myDeepRL = DeepRLToy(myEnvi, myNN)

def step_callback():
	return

def render_callback():
	gl_render.render_ground(color=[1.0,1.0,1.0],axis='z')
	myEnvi.detective.render(color=[0,0,1])
	myEnvi.criminal.render(color=[1,0,0])
	if flag['Train']:
		myDeepRL.run(50, 50)

def keyboard_callback(key):
	if key == 'r':
		myEnvi.set_random_state()
	elif key == 't':
		myEnvi.step_forward(dt, vel_random())
	elif key == ' ':
		state = myEnvi.state()
		action = myDeepRL.get_action(state)
		qvalue = myDeepRL.get_qvalue(state)
		reward = myEnvi.step_forward(dt, action)
		print 'S:', state, 'A:', action, 'R:', reward, 'Q:', qvalue
		if myEnvi.check_collision(myEnvi.detective.pos):
			for i in range(20):
				print 'COLLISION',
			print ' '
	elif key == 'd':
		flag['Train'] = not flag['Train']
	elif key == '0':
		sample_idx = myDeepRL.replay_buffer['critic'].sample_idx(10)
		data = myDeepRL.sample('critic', sample_idx)
		print data

pydart.glutgui.glutgui_base.run(title='example_toy',
						trans=[0, 0, -30],
						keyboard_callback=keyboard_callback,
						render_callback=render_callback)