import nn
import tensorflow as tf

class MyNNSimple(nn.NNBase):
	def __init__(self, name):
		nn.NNBase.__init__(self, name)
		self.dropout_keep_prob = 0.5
		self.train_a = None
		self.train_q = None
		self.train_sum = None
		self.eval_q = None
		self.eval_a = None
		self.placeholder_sensor = None
		self.placeholder_skel = None
		self.placeholder_target_qvalue = None
		self.placeholder_target_action = None
		self.placeholder_dropout_keep_prob = None
		self.loss_q = None
		self.loss_a = None
		self.loss_sum = None
	def initialize(self, data):
		tf.reset_default_graph()
		with self.graph.as_default():
			s = data[0] 	# sensor dimension
			d = data[1]		# state dimension
			a = data[2]		# action dimension
			# 
			state_sensor = tf.placeholder(tf.float32, [None,s])
			state_skel = tf.placeholder(tf.float32, [None,d])
			target_qvalue = tf.placeholder(tf.float32, [None,1])
			target_action = tf.placeholder(tf.float32, [None,a])
			keep_prob = tf.placeholder(tf.float32)

			# 1st layer : the sensor and the skel are combined
			h_comb1 = tf.concat(1, [state_sensor, state_skel])
			W_fc1 = nn.weight_variable([s+d, 32])
			b_fc1 = nn.bias_variable([32])
			h_fc1 = tf.nn.relu(tf.matmul(h_comb1, W_fc1) + b_fc1)
			# 2nd layer
			W_fc2 = nn.weight_variable([32, 32])
			b_fc2 = nn.bias_variable([32])
			h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
			# 3rd layer
			W_fc3 = nn.weight_variable([32, 16])
			b_fc3 = nn.bias_variable([16])
			h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
			h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
			# Layer for Q value
			W_qvalue = nn.weight_variable([16, 1])
			b_qvalue = nn.bias_variable([1])
			h_qvalue = tf.matmul(h_fc3_drop, W_qvalue) + b_qvalue
			# Layer for action
			W_action = nn.weight_variable([16, a])
			b_action = nn.bias_variable([a])
			h_action = tf.matmul(h_fc3_drop, W_action) + b_action

			# Optimizer
			loss_qvalue = tf.mul(100.0,tf.reduce_mean(tf.square(target_qvalue - h_qvalue)))
			loss_action = tf.mul(100.0,tf.reduce_mean(tf.square(target_action - h_action)))
			loss_sum = tf.add(loss_qvalue, loss_action)
			# Trainning
			# self.train_q = tf.train.AdamOptimizer(1e-4).minimize(loss_qvalue)
			# self.train_a = tf.train.AdamOptimizer(1e-4).minimize(loss_action)
			# self.train_sum = tf.train.AdamOptimizer(1e-4).minimize(loss_sum)
			self.train_q = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_qvalue)
			self.train_a = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_action)
			self.train_sum = tf.train.GradientDescentOptimizer(1e-4).minimize(loss_sum)
			# Evaultion
			self.eval_q = h_qvalue
			self.eval_a = h_action
			# Place holders
			self.placeholder_sensor = state_sensor
			self.placeholder_skel = state_skel
			self.placeholder_target_qvalue = target_qvalue
			self.placeholder_target_action = target_action
			self.placeholder_dropout_keep_prob = keep_prob
			# Loss
			self.loss_q = loss_qvalue
			self.loss_a = loss_action
			self.loss_sum = loss_sum
			# Initialize all variables
			self.sess = tf.Session(graph=self.graph)
			self.sess.run(tf.initialize_all_variables())
			self.saver = tf.train.Saver()
			self.initialized = True
	def train(self, data):
		# data_state_sensor = data[0]
		# data_state_skel = data[1]
		# target_qvalue = data[2]
		# target_action = data[3]
		# self.train_qvalue([\
		# 	data_state_sensor,data_state_skel,target_qvalue])
		# self.train_action([\
		# 	data_state_sensor,data_state_skel,target_action])
		self.sess.run(self.train_sum, feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_target_qvalue: data[2],
			self.placeholder_target_action: data[3],
			self.placeholder_dropout_keep_prob: self.dropout_keep_prob})
	def eval(self, data):
		q = self.eval_qvalue(data)
		a = self.eval_action(data)
		return q, a
	def loss(self, data):
		data_state_sensor = data[0]
		data_state_skel = data[1]
		target_qvalue = data[2]
		target_action = data[3]
		q = self.loss_qvalue([data_state_sensor, data_state_skel, target_qvalue])
		a = self.loss_action([data_state_sensor, data_state_skel, target_action])
		return q, a
	def eval_qvalue(self, data):
		val = self.sess.run(self.eval_q, feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def eval_action(self, data):
		val = self.sess.run(self.eval_a, feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def loss_qvalue(self, data):
		val = self.sess.run(self.loss_q,feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_target_qvalue: data[2],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def loss_action(self, data):
		val = self.sess.run(self.loss_a,feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_target_action: data[2],
			self.placeholder_dropout_keep_prob: 1.0})
		return val
	def train_qvalue(self, data):
		self.sess.run(self.train_q, feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_target_qvalue: data[2],
			self.placeholder_dropout_keep_prob: self.dropout_keep_prob})
	def train_action(self, data):
		self.sess.run(self.train_a, feed_dict={
			self.placeholder_sensor: data[0],
			self.placeholder_skel: data[1],
			self.placeholder_target_action: data[2],
			self.placeholder_dropout_keep_prob: self.dropout_keep_prob})