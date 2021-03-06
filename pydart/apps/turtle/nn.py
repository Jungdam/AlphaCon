from abc import ABCMeta, abstractmethod
import tensorflow as tf
import warnings
import datetime
import numpy as np


def weight_variable(name, shape):
    # initial = tf.truncated_normal(shape, stddev=0.2)
    # return tf.Variable(initial)
    xavier = tf.contrib.layers.xavier_initializer()
    return tf.get_variable(name, shape=shape, initializer=xavier)


def bias_variable(name, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


class Normalizer:
    def __init__(self, dim):
        self.dim = dim
        self.mean = np.zeros(dim)
        self.sigma = np.ones(dim)
        self.epsilon = 1.0e-5
    def raw_to_norm(self, data):
        z = []
        for datum in data:
            z.append((datum-self.mean)/(self.sigma+self.epsilon))
        return z
    def norm_to_raw(self, data):
        d = []
        for datum in data:
            d.append(self.mean + datum*(self.sigma+self.epsilon))


class Variables:
    def __init__(self):
        self.var = {}

    def get_variable(self, name, from_copy=False):
        if from_copy:
            return self.var[name][1]
        else:
            return self.var[name][0]

    def weight_variable(self, name, shape):
        var = weight_variable(name, shape)
        self.var[name] = [var, None, None]
        return var

    def bias_variable(self, name, shape):
        var = bias_variable(name, shape)
        self.var[name] = [var, None, None]
        return var

    def copy_all(self):
        for key, value in self.var.iteritems():
            var = value[0]
            var_copy = tf.Variable(var.initialized_value(), name=key + '_copy')
            value[1] = var_copy
            value[2] = var_copy.assign(var)

    def copy(self, name):
        value = self.var[name]
        var = value[0]
        var_copy = tf.Variable(var.initialized_value(), name=name + '_copy')
        value[1] = var_copy
        value[2] = var_copy.assign(var)
        return var_copy

    def save(self, sess):
        for key, value in self.var.iteritems():
            op = value[2]
            sess.run(op)

    def _print(self):
        for key, value in self.var.iteritems():
            print value


class Layer:
    def __init__(self, name, var, var_copied=False,
        tensor_in=None, dim_in=0, dim_out=0,
        act_fn=tf.nn.relu, dropout_enabled=False, dropout_placeholder=None):
        self.name = name
        self.var = var
        self.var_copied = var_copied
        self.W = None
        self.b = None
        self.h = None
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.act_fn = act_fn
        self.dropout_enabled = dropout_enabled
        self.dropout_placeholder = dropout_placeholder
        self.initialize(tensor_in)

    def initialize(self, tensor_in):
        if self.var_copied:
            self.W = self.var.copy(self.name + '_W')
            self.b = self.var.copy(self.name + '_b')
        else:
            self.W = self.var.weight_variable(self.name + '_W', [self.dim_in, self.dim_out])
            self.b = self.var.bias_variable(self.name + '_b', [self.dim_out])
        if self.act_fn is not None:
            h = self.act_fn(tf.matmul(tensor_in, self.W) + self.b)
        else:
            h = tf.matmul(tensor_in, self.W) + self.b
        if self.dropout_enabled and self.dropout_placeholder is not None:
            h = tf.nn.dropout(h, self.dropout_placeholder)
        else:
            self.dropout_enabled = False
        self.h = h

    def copy(self, tensor_in):
        return Layer(self.name, self.var, True,
            tensor_in, self.dim_in, self.dim_out, self.act_fn)


class NNBase:
    __metaclass__ = ABCMeta

    def __init__(self, name):
        self.name = self.unique_name(name)
        self.graph = tf.Graph()
        self.initialized = False
        self.sess = None
        self.saver = None
        self.var = Variables()
        self.writer = None
        self.merged = None

    @abstractmethod
    def initialize(self):
        """
        Define the neural network graph
        """
        raise NotImplementedError("Must override")
    # @abstractmethod
    # # Run trainning session for given input/output pairs
    # def train(self, data):
    #     raise NotImplementedError("Must override")
    # @abstractmethod
    # # Evaluate the network for given input
    # def eval(self, data):
    #     raise NotImplementedError("Must override")
    # # Compute loss value for given test data
    # @abstractmethod
    # def loss(self, data):
    #     raise NotImplementedError("Must override")

    def get_name(self):
        return self.name

    def unique_name(self, name):
        date = datetime.date.today()
        d = datetime.datetime.today()
        return name + d.strftime("_%Y%m%d_%H%M%S")

    def save_file(self, file_name=None):
        if file_name is None:
            file_name = self.name
        self.saver.save(self.sess, file_name)
        print '[NeuralNet] model saved', file_name

    def load_file(self, file_name=None):
        if file_name is None:
            file_name = self.name
        self.saver.restore(self.sess, file_name)
        print '[NeuralNet] model loaded:', file_name

    def summary(self):
        summary_op = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

    def save(self, dir):
        save_path = self.saver.save(self.sess, dir + self.name + '.ckpt')
        print("Model saved in file: %s" % save_path)

    def restore(self, file):
        if file is None:
            return False
        ckpt = tf.train.get_checkpoint_state(file)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print '[ NeuralNet ]', 'Checkpoint', ckpt.model_checkpoint_path, 'is loaded'
            return True
        else:
            print '[ NeuralNet ]', 'Checkpoint', file, 'is invalid'
            return False

# class MyNN(NNBase):
#     def __init__(self, name):
#         NNBase.__init__(self, name)
#         self.dropout_keep_prob = 0.5
#         self.train_a = None
#         self.train_q = None
#         self.eval_q = None
#         self.eval_a = None
#         self.placeholder_eye = None
#         self.placeholder_skel = None
#         self.placeholder_target_qvalue = None
#         self.placeholder_target_action = None
#         self.placeholder_dropout_keep_prob = None
#         self.loss_q = None
#         self.loss_a = None
#     def initialize(self, data):
#         tf.reset_default_graph()
#         with self.graph.as_default():
#             w,h = data[0]     # image dimension
#             d = data[1]        # state dimension
#             a = data[2]        # action dimension
#             print w, h, d, a
#             # 
#             state_eye = tf.placeholder(tf.float32, [None,w,h,1])
#             state_skel = tf.placeholder(tf.float32, [None,d])
#             target_qvalue = tf.placeholder(tf.float32, [None,1])
#             target_action = tf.placeholder(tf.float32, [None,a])
#             keep_prob = tf.placeholder(tf.float32)
            
#             #
#             # Max Pool Model
#             #
#             # # Frist conv layer for the eye
#             # W_conv1 = weight_variable([5, 5, 1, 32])
#             # b_conv1 = bias_variable([32])
#             # h_conv1 = tf.nn.relu(conv2d(state_eye, W_conv1) + b_conv1)
#             # h_pool1 = max_pool_2x2(h_conv1)
#             # # Second conv layer for the eye
#             # W_conv2 = weight_variable([5, 5, 32, 64])
#             # b_conv2 = bias_variable([64])
#             # h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#             # h_pool2 = max_pool_2x2(h_conv2)
#             # # Fully connected layer for the eye
#             # W_fc1 = weight_variable([(w/4)*(h/4)*64, 256])
#             # b_fc1 = bias_variable([256])
#             # h_pool2_flat = tf.reshape(h_pool2, [-1, (w/4)*(h/4)*64])
#             # h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
            
#             #
#             # No Pool Model
#             #
#             # Frist conv layer for the eye
#             W_conv1 = weight_variable([4, 4, 1, 32])
#             b_conv1 = bias_variable([32])
#             h_conv1 = tf.nn.relu(conv2d(state_eye, W_conv1) + b_conv1)
#             # Second conv layer for the eye
#             W_conv2 = weight_variable([2, 2, 32, 64])
#             b_conv2 = bias_variable([64])
#             h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
#             # Fully connected layer for the eye
#             W_fc1 = weight_variable([w*h*64, 512])
#             b_fc1 = bias_variable([512])
#             h_covn2_flat = tf.reshape(h_conv2, [-1, w*h*64])
#             h_fc1 = tf.nn.relu(tf.matmul(h_covn2_flat, W_fc1) + b_fc1)

#             # Combined layer for the eye and the skel
#             W_fc2 = weight_variable([512+d, 1024])
#             b_fc2 = bias_variable([1024])
#             h_comb1 = tf.concat(1, [h_fc1, state_skel])
#             h_fc2 = tf.nn.relu(tf.matmul(h_comb1, W_fc2) + b_fc2)
#             h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)
#             #
#             W_fc3_qvalue = weight_variable([1024, 1])
#             b_fc3_qvalue = bias_variable([1])
#             h_fc3_qvalue = tf.matmul(h_fc2_drop, W_fc3_qvalue) + b_fc3_qvalue
#             W_fc3_action = weight_variable([1024, a])
#             b_fc3_action = bias_variable([a])
#             h_fc3_action = tf.matmul(h_fc2_drop, W_fc3_action) + b_fc3_action
#             # Optimizer
#             loss_qvalue = tf.reduce_mean(100.0*tf.square(target_qvalue - h_fc3_qvalue))
#             loss_action = tf.reduce_mean(100.0*tf.square(target_action - h_fc3_action))
#             # Trainning
#             self.train_q = tf.train.AdamOptimizer(1e-4).minimize(loss_qvalue)
#             self.train_a = tf.train.AdamOptimizer(1e-4).minimize(loss_action)
#             # self.train_q = tf.train.GradientDescentOptimizer(0.001).minimize(loss_qvalue)
#             # self.train_a = tf.train.GradientDescentOptimizer(0.001).minimize(loss_action)
#             # Evaultion
#             self.eval_q = h_fc3_qvalue
#             self.eval_a = h_fc3_action
#             # Place holders
#             self.placeholder_eye = state_eye
#             self.placeholder_skel = state_skel
#             self.placeholder_target_qvalue = target_qvalue
#             self.placeholder_target_action = target_action
#             self.placeholder_dropout_keep_prob = keep_prob
#             # Loss
#             self.loss_q = loss_qvalue
#             self.loss_a = loss_action
#             # Initialize all variables
#             self.sess = tf.Session(graph=self.graph)
#             self.sess.run(tf.initialize_all_variables())
#             self.saver = tf.train.Saver()
#             self.initialized = True
#     def train(self, data):
#         data_state_eye = data[0]
#         data_state_skel = data[1]
#         target_qvalue = data[2]
#         target_action = data[3]
#         self.train_qvalue([\
#             data_state_eye,data_state_skel,target_qvalue])
#         self.train_action([\
#             data_state_eye,data_state_skel,target_action])
#     def eval(self, data):
#         q = self.eval_qvalue(data)
#         a = self.eval_action(data)
#         return q, a
#     def loss(self, data):
#         data_state_eye = data[0]
#         data_state_skel = data[1]
#         target_qvalue = data[2]
#         target_action = data[3]
#         q = self.loss_qvalue([data_state_eye, data_state_skel, target_qvalue])
#         a = self.loss_action([data_state_eye, data_state_skel, target_action])
#         return q, a
#     def eval_qvalue(self, data):
#         val = self.sess.run(self.eval_q, feed_dict={
#             self.placeholder_eye: data[0],
#             self.placeholder_skel: data[1],
#             self.placeholder_dropout_keep_prob: 1.0})
#         return val
#     def eval_action(self, data):
#         val = self.sess.run(self.eval_a, feed_dict={
#             self.placeholder_eye: data[0],
#             self.placeholder_skel: data[1],
#             self.placeholder_dropout_keep_prob: 1.0})
#         return val
#     def loss_qvalue(self, data):
#         val = self.sess.run(self.loss_q,feed_dict={
#             self.placeholder_eye: data[0],
#             self.placeholder_skel: data[1],
#             self.placeholder_target_qvalue: data[2],
#             self.placeholder_dropout_keep_prob: 1.0})
#         return val
#     def loss_action(self, data):
#         val = self.sess.run(self.loss_a,feed_dict={
#             self.placeholder_eye: data[0],
#             self.placeholder_skel: data[1],
#             self.placeholder_target_action: data[2],
#             self.placeholder_dropout_keep_prob: 1.0})
#         return val
#     def train_qvalue(self, data):
#         self.sess.run(self.train_q, feed_dict={
#             self.placeholder_eye: data[0],
#             self.placeholder_skel: data[1],
#             self.placeholder_target_qvalue: data[2],
#             self.placeholder_dropout_keep_prob: self.dropout_keep_prob})
#     def train_action(self, data):
#         self.sess.run(self.train_a, feed_dict={
#             self.placeholder_eye: data[0],
#             self.placeholder_skel: data[1],
#             self.placeholder_target_action: data[2],
#             self.placeholder_dropout_keep_prob: self.dropout_keep_prob})

