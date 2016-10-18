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
import pickle
import profile
import multiprocessing as mp
import time
import action as ac
import eye


np.set_printoptions(precision=3)
flag = {}
flag['Train'] = False
cnt_target_update = 0
max_target_update = 8
log_dir = './data/tensorflow/log'
ckpt_load_dir = None
# ckpt_load_dir = './data/tensorflow/model/torque'
ckpt_save_dir = './data/tensorflow/model/torque'
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle_test.skel'
warmup_file = None
# warmup_file = './data/warmup/0.2_100_10_torque.warmup'

elapsed_time = 0.0
num_init_wingbeat = 0
dt = 1.0 / 1000.0
max_client = 8
max_steps = 500

gl_init = False

pydart.init()
print('pydart initialization OK')

profile = profile.Profile()
lock = mp.Lock()

def preprocess_state(state):
    return state
    s = np.array(state)
    for i in range(len(s)):
        v = s[i]
        if not -math.pi < v < math.pi:
            v = v % 2 * math.pi
            v -= math.pi
            s[i] = v
    return s


def gen_warmup_data_one_process(q, idx, sigma, num_episode, num_wingbeat, sampling_time):
    env = EnvPoseDriven(dt, skel_file, 2)
    con = env.skel.controller
    act = ActionPoseDirven(13, idx)
    data = []
    for ep in xrange(num_episode):
        env.reset()
        for i in xrange(num_wingbeat):
            action = act.random(sigma)
            con.add_action(action)
            elapsed_time = 0.0
            while True:
                state_skel_init = np.array(env.skel.states())
                env.step_forward()
                elapsed_time += dt
                if elapsed_time > sampling_time:
                    elapsed_time = 0.0
                    state_skel_term = np.array(env.skel.states())
                    torque = np.array(con.get_tau()[6:])
                    data.append([state_skel_init, state_skel_term, torque])
                    if len(data) % 1000 == 0:
                        print '[', idx, ep, ']', len(data), 'data generated'
                if con.is_new_wingbeat():
                    break
    q.put(data)


def gen_warmup_data(sigma, num_episode=2500, num_wingbeat=20, sampling_time=0.1, write_file=True):
    num_cores = mp.cpu_count()
    data = []
    ps = []
    q = mp.Queue()
    num_episode_assigned = [num_episode / num_cores] * num_cores
    for i in range(num_episode % num_cores):
        num_episode_assigned[i] += 1
    num_active_cores = 0
    for i in range(num_cores):
        if num_episode_assigned[i] > 0:
            num_active_cores += 1
    for i in range(num_active_cores):
        p = mp.Process(
            target=gen_warmup_data_one_process,
            args=(q, i, sigma, num_episode_assigned[i], num_wingbeat, sampling_time))
        p.start()
        ps.append(p)
        time.sleep(0.1)
    for i in range(num_active_cores):
        d = q.get()
        data = data + d
    for p in ps:
        p.join()
    if write_file:
        f = open(str(sigma[0]) + '_' + str(num_episode) +
                 '_' + str(num_wingbeat) + '_torque.warmup', 'w')
        pickle.dump(data, f)
        f.close()
    return data


class Action(ac.ActionBase):
    def initialize(self):
        self.val_def = np.zeros(self.dim)
        self.val_min = -500.0 * np.ones(self.dim)
        self.val_max = 500.0 * np.ones(self.dim)


class Target:
    def __init__(
            self,
            base=np.array([0.0, 1.0, 0.0]),
            offset=np.array([0.0, 0.0, 3.0]),
            sigma=np.array([1.0, 1.0, 1.0]),
            sigma_lo=np.array([-2.0, -2.0, -3.0]),
            sigma_hi=np.array([2.0, 2.0, 3.0])):
        self.base = base
        self.offset = offset
        self.sigma = sigma
        self.sigma_lo = sigma_lo
        self.sigma_hi = sigma_hi
        self.pos = self.new_pos()

    def reset(self):
        self.pos = self.new_pos()

    def new_pos(self):
        r = np.random.normal(np.zeros(3), self.sigma)
        for i in range(3):
            r[i] = basics.clamp(r[i], self.sigma_lo[i], self.sigma_hi[i])
        return self.base + self.offset + r

    def get_pos(self):
        return self.pos

    def set_pos(self, pos):
        self.pos = pos


class Env(env.EnvironmentBase):
    def __init__(self, dt, skel_file):
        self.world = pydart.create_world(dt, skel_file)
        self.skel = self.world.skels[0]
        self.skel.controller = controller.ControllerTorque(
            self.world, self.skel)
        self.target = Target()
        self.world.step()
        self.world.push()
        self.world.push()
        self.reset()

        # self.save_eye_image('test' + str(np.random.randint(15)) + '.png')

        # pose_lo = self.skel.q_lo
        # pose_hi = self.skel.q_hi
        # pose = self.skel.q
        # p = 0.0
        # for i in range(self.skel.ndofs):
        #     diff_lo = pose[i] - pose_lo[i]
        #     diff_hi = pose_hi[i] - pose[i]
        #     if diff_lo < 0.0:
        #         p += diff_lo * diff_lo
        #     if diff_hi < 0.0:
        #         p += diff_hi * diff_hi
        # print pose
        # print pose_lo
        # print pose_hi
        # print p

    def reset(self):
        self.world.reset()
        self.skel.controller.reset()
        self.target.reset()

    def goodness(self, verbose=False):
        R, p = mmMath.T2Rp(self.skel.body('trunk').T)
        l = np.linalg.norm(self.target.get_pos() - p)
        goal = math.exp(-0.5 * l * l)
        goal_w = 1.0e00

        t = np.linalg.norm(self.skel.controller.get_torque())
        effort = t * t
        effort_w = -1.0e-5

        pose_lo = self.skel.q_lo
        pose_hi = self.skel.q_hi
        pose = self.skel.q
        pose_limit = 0.0
        for i in range(self.skel.ndofs):
            diff_lo = pose[i] - pose_lo[i]
            diff_hi = pose_hi[i] - pose[i]
            if diff_lo < 0.0:
                pose_limit += diff_lo * diff_lo
            if diff_hi < 0.0:
                pose_limit += diff_hi * diff_hi
        pose_limit_w = -1.0e-02

        all_sum = goal_w * goal + effort_w * effort + pose_limit_w * pose_limit

        if verbose:
            print '---------------------------------------------------'
            print 'Goal(', goal, goal_w * goal, ')',
            print 'Effort(', effort, effort_w * effort, ')',
            print 'PoseLimit(', pose_limit, pose_limit_w * pose_limit, ')',
            print all_sum
            print '---------------------------------------------------'

        # effort = math.exp(-1.0e-2 * t * t)
        # print "Goal:", goal, "Effort:", effort
        return all_sum

    def state(self):
        # State of skeleton
        state_skel = []
        # prepare for computing local coordinate of the trunk
        body_trunk = self.skel.body('trunk')
        R_trunk, p_trunk = mmMath.T2Rp(body_trunk.T)
        R_trunk_inv = np.linalg.inv(R_trunk)
        # trunk state
        vel_trunk = body_trunk.world_com_velocity()
        state_skel.append(np.dot(R_trunk_inv, vel_trunk))
        # other bodies state
        bodies = []
        bodies = ['left_arm', 'right_arm', 'left_hand',
                  'right_hand', 'left_leg', 'right_leg']
        # bodies = ['left_arm', 'right_arm']
        for name in bodies:
            body = self.skel.body(name)
            l = body.world_com() - p_trunk
            v = body.world_com_velocity()
            state_skel.append(np.dot(R_trunk_inv, l))
            state_skel.append(np.dot(R_trunk_inv, v))
        state_skel = np.array(state_skel).flatten()

        # s = np.array(self.skel.states())
        # p = preprocess_state(s[6:self.skel.ndofs])
        # v = s[self.skel.ndofs+6:]
        # state_skel = np.hstack([p, v])

        # State of sensor
        #
        state_sensor = np.dot(R_trunk_inv, self.target.get_pos() - p_trunk)
        return np.hstack([state_skel, state_sensor])

    def step_forward(self, apply_controller=True, apply_aero=True):
        self.world.step(apply_controller, apply_aero)
        return self.goodness()

    def step(self, torque=None):
        if torque is not None:
            self.skel.controller.set_torque(torque)
        self.world.step(True, True)
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
        glTranslated(p[0], p[1], p[2])
        glutSolidSphere(0.25, 10, 10)
        glPopMatrix()

    def eye_render(self):
        # glDisable(GL_LIGHTING)
        glColor3d(1.0, 0.0, 0.0)
        p = self.target.get_pos()
        glPushMatrix()
        glTranslated(p[0], p[1], p[2])
        glutSolidSphere(0.25, 10, 10)
        glPopMatrix()
        print 'eye rendered'

    # def save_eye_image(self, file_name):
    #     lock.acquire()
    #     print self.skel.body('trunk').T
    #     print self.target.get_pos()
    #     self.eye.update(self.skel.body('trunk').T)
    #     lock.release()
    #     self.eye.save_image(file_name)

class ActionPoseDirven(ac.ActionBase):
    def initialize(self):
        self.val_def = np.array([
            -1.25567921, 0.6118376, 0.53513041,
            0.28105493, 0.78491477, -0.65140349,
            -1.25567921, 0.6118376, -0.53513041,
            -0.28105493, -0.78491477, 0.65140349,
            1.5])
        self.val_min = np.array([
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
            1.0])
        self.val_max = np.array([
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
            2.0])


class EnvPoseDriven(env.EnvironmentBase):
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
                break
            self.world.step()

    def reset(self):
        self.world.reset()
        self.skel.controller.reset()
        self.target.reset()

    def goodness(self):
        R, p = mmMath.T2Rp(self.skel.body('trunk').T)
        l = np.linalg.norm(self.target.get_pos() - p)
        return math.exp(-0.5 * l * l)

    def state(self):
        #
        # State of skeleton
        #
        state_skel = []
        # prepare for computing local coordinate of the trunk
        body_trunk = self.skel.body('trunk')
        R_trunk, p_trunk = mmMath.T2Rp(body_trunk.T)
        R_trunk_inv = np.linalg.inv(R_trunk)
        # trunk state
        vel_trunk = body_trunk.world_com_velocity()
        state_skel.append(np.dot(R_trunk_inv, vel_trunk))
        # other bodies state
        bodies = []
        # bodies = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
        bodies = ['left_arm', 'right_arm']
        for name in bodies:
            body = self.skel.body(name)
            l = body.world_com() - p_trunk
            v = body.world_com_velocity()
            state_skel.append(np.dot(R_trunk_inv, l))
            state_skel.append(np.dot(R_trunk_inv, v))
        state_skel = np.array(state_skel).flatten()
        #
        # State of sensor
        #
        state_sensor = np.dot(R_trunk_inv, self.target.get_pos() - p_trunk)
        # return state_sensor
        return np.hstack([state_skel, state_sensor])

    def step_forward(self, apply_controller=True, apply_aero=True):
        self.world.step(apply_controller, apply_aero)
        return self.goodness()

    def step(self, wingbeat=None):
        if wingbeat is not None:
            self.skel.controller.add_action(wingbeat)
        while True:
            self.world.step(True, True)
            if self.skel.controller.is_new_wingbeat():
                break
        # print proc_name, 'end'
        return self.goodness()

    def render(self):
        return


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
        elif signal == "set_world_state":
            self.env.world.set_states(data)
        elif signal == "get_world_state":
            s = self.env.world.states()
            self.q_result.put(s)


class En_Master_Custom(env.Environment_Master):
    def get_target_pos(self):
        return self.run(self.num_slave * ["get_target_pos"])

    def set_target_pos(self, pos):
        self.run(self.num_slave * ["set_target_pos"], pos, return_data=False)

    def get_world_state(self):
        return self.run(self.num_slave * ["get_world_state"])

    def set_world_state(self, state):
        self.run(self.num_slave * ["set_world_state"], state, return_data=False)


def gen_env(args):
    dt = args[0]
    skel_file = args[1]
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
        self.learning_rate = 1.0 * 1e-3
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
            state = tf.placeholder(tf.float32, [None, d])
            target_qvalue = tf.placeholder(tf.float32, [None, 1])
            target_action = tf.placeholder(tf.float32, [None, a])
            keep_prob = tf.placeholder(tf.float32)

            # Network definition
            layer_1 = nn.Layer('layer_1', self.var, False, state, d, 64)
            layer_2 = nn.Layer('layer_2', self.var, False, layer_1.h, 64, 64)
            layer_3 = nn.Layer('layer_3', self.var, False, layer_2.h, 64, 64,
                                dropout_enabled=True, dropout_placeholder=keep_prob)
            layer_q = nn.Layer('layer_q', self.var, False,
                               layer_3.h, 64, 1, None)
            layer_a = nn.Layer('layer_a', self.var, False,
                               layer_3.h, 64, a, None)

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
            # self.learning_rate, global_step, 1000000, 0.96, staircase=True)

            # Optimization for trainning
            # opt_q = tf.train.GradientDescentOptimizer(self.learning_rate)
            # opt_a = tf.train.GradientDescentOptimizer(self.learning_rate)
            opt_q = tf.train.AdamOptimizer(self.learning_rate)
            opt_a = tf.train.AdamOptimizer(self.learning_rate)
            grads_and_vars_q = opt_q.compute_gradients(loss_qvalue)
            grads_and_vars_a = opt_a.compute_gradients(loss_action)

            # Gradient clipping
            capped_grads_and_vars_q = [
                (tf.clip_by_norm(grad, 100.0), var) if grad is not None
                else (None, var) for grad, var in grads_and_vars_q]
            capped_grads_and_vars_a = [
                (tf.clip_by_norm(grad, 100.0), var) if grad is not None
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
        val = self.sess.run(self.loss_q, feed_dict={
            self.placeholder_state: data[0],
            self.placeholder_target_qvalue: data[1],
            self.placeholder_dropout_keep_prob: 1.0})
        return val

    def loss_action(self, data):
        val = self.sess.run(self.loss_a, feed_dict={
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
    def __init__(self, env, nn, ac, warmup_file=None):
        deepRL.DeepRLBase.__init__(self, warmup_file)
        self.exp_prob_default = 0.5
        self.exp_noise_default = 10.0
        self.qvalue_knoll_default = 0.7
        self.replay_buffer['actor'] = deepRL.ReplayBuffer(500000)
        self.replay_buffer['critic'] = deepRL.ReplayBuffer(500000)
        self.env = env
        self.nn = nn
        self.ac = ac
        self.train_iter = 10
        self.warmup_size = 0
        self.max_data_gen = 20000000
        self.sample_size = 128
        self.target_pos_pool = []
        warmup_data = []
        while len(self.target_pos_pool) < self.env.num_slave:
            self.env.reset()
            self.target_pos_pool += self.env.get_target_pos()
        if warmup_file is None:
            print '[DeepRL]', 'generating warmup data ...'
            warmup_data = self.convert_warmup_file_to_buffer_data(
                Env(dt, skel_file),
                gen_warmup_data([0.1] * 13, 30, 10, 0.01, False))
            print len(warmup_data), 'data were generated'
        else:
            print '[DeepRL]', 'loading warmup file ...'
            warmup_data = self.convert_warmup_file_to_buffer_data(
                Env(dt, skel_file), None, self.warmup_file)
            print len(warmup_data), 'data were loaded'
        self.replay_buffer['actor'].append(warmup_data, verbose=True)
        self.replay_buffer['critic'].append(warmup_data, verbose=True)
        self.warmup_size = self.replay_buffer['actor'].size_accum
        num_action_trained = 0
        for i in range(1000):
            self.train_action(self.sample_size, False)
        self.save_variables()
        for i in range(2000):
            self.train_qvalue(self.sample_size)
            num_action_trained += self.train_action(self.sample_size)
            if i % 20 == 0:
                self.save_variables()
            if i % 100 == 0:
                self.print_loss()
                print 'action_trained:', num_action_trained
                num_action_trained = 0
        self.save_variables()

    def convert_warmup_file_to_buffer_data(
            self, env, data=None, file_name=None):
        if data is None:
            f = open(file_name, 'r')
            data = pickle.load(f)
        size = len(data)
        tuples = []
        cnt = 0
        for d in data:
            state_skel_init = d[0]
            state_skel_term = d[1]
            action = d[2]

            env.reset()

            env.world.skel.set_states(state_skel_init)
            state_init = env.state()

            env.world.skel.set_states(state_skel_term)
            state_term = env.state()

            reward = env.goodness()
            action_delta = self.ac.delta(action)

            world_state = state_skel_term

            t = [state_init, action_delta, [reward], state_term, world_state]
            if basics.check_valid_data(t):
                tuples.append(t)
                cnt += 1
                if cnt % 5000 == 0:
                    print cnt, ' data were processed'
        return tuples

    def run(self, max_steps=32, verbose=True):
        if self.is_finished_trainning():
            return
        num_action_trained = 0
        self.init_step()
        for i in range(max_steps):
            # print '[', i, 'steps ] start',
            buffer_names, data = self.step()
            for j in range(len(data)):
                if data[j] is None:
                    self.env.reset(j)
                else:
                    self.replay_buffer[buffer_names[j]].append([data[j]])
            if not self.is_warming_up():
                self.train_qvalue(self.sample_size)
                num_action_trained += self.train_action(self.sample_size)
            # print 'end'
        # Print statistics
        if verbose:
            print '[ stats ]', ' warmup: ', self.is_warming_up(),
            for buffer_name in self.replay_buffer.keys():
                print '[', buffer_name,
                print self.replay_buffer[buffer_name].size,
                print self.replay_buffer[buffer_name].size_accum, ']',
            if not self.is_warming_up():
                self.print_loss()
            print ', action_trained:', num_action_trained

    def measure_controller_quality(self, max_steps):
        # num_slave = self.env.num_slave
        # num_iter = len(self.target_pos_pool)/num_slave
        # cnt_wingbeat = 0
        # sum_reward = 0.0
        # for i in range(num_iter):
        #     self.env.set_target_pos(self.target_pos_pool[num_slave*i:num_slave*(i+1)])
        #     for j in range(max_steps):
        #         buffer_names, data = self.step(force_critic=True)
        #         for k in range(len(data)):
        #             if data[k] is None:
        #                 continue
        #             cnt_wingbeat += 1
        #             sum_reward += data[k][2][0]
        # return sum_reward/cnt_wingbeat
        return 0.0

    def init_step(self):
        self.env.reset()
        sample_idx = self.replay_buffer[
            'critic'].sample_idx(self.env.num_slave)
        data = self.sample('critic', sample_idx)
        self.env.set_world_state(data[4])

    def step(self):
        state_inits = self.env.state()
        actions = self.get_actions(state_inits)
        for a in actions:
            a = self.ac.clamp(a)
        buffer_names = []
        for i in range(self.env.num_slave):
            buffer_name = 'critic'
            if self.determine_exploration():
                actions[i] += np.random.normal(
                    np.zeros(self.ac.dim),
                    self.get_exploration_noise() * np.ones(self.ac.dim))
                buffer_name = 'actor'
            actions[i] = self.ac.clamp(actions[i])
            buffer_names.append(buffer_name)
        rewards = self.env.step(actions)
        state_terms = self.env.state()
        world_states = self.env.get_world_state()
        tuples = []
        for i in range(self.env.num_slave):
            act = self.ac.delta(actions[i])
            t = [state_inits[i], act, [rewards[i]],
                 state_terms[i], world_states[i]]
            if basics.check_valid_data(t):
                tuples.append(t)
            else:
                tuples.append(None)
        return buffer_names, tuples

    def sample(self, buffer_name, idx):
        if not idx:
            raise Exception('DeepRL:sample', 'no index')
        data_state = []
        data_action = []
        data_reward = []
        data_state_prime = []
        data_world_state = []
        for i in idx:
            datum = self.replay_buffer[buffer_name].data[i]
            data_state.append(datum[0])
            data_action.append(datum[1])
            data_reward.append(datum[2])
            data_state_prime.append(datum[3])
            data_world_state.append(datum[4])
        return [
            np.array(data_state),
            np.array(data_action),
            np.array(data_reward),
            np.array(data_state_prime),
            np.array(data_world_state)]

    def compute_target_qvalue(self, reward, state_prime):
        qvalue_prime = self.nn.eval_qvalue([state_prime], True)
        target_qvalue = reward + self.discount_factor * qvalue_prime
        return target_qvalue

    def train_qvalue(self, sample_size, verbose=False):
        sample_idx = self.replay_buffer['critic'].sample_idx(sample_size)
        data = self.sample('critic', sample_idx)
        if data:
            data_state = data[0]
            data_reward = data[2]
            data_state_prime = data[3]
            target_qvalue = self.compute_target_qvalue(
                data_reward, data_state_prime)
            for i in range(self.train_iter):
                self.nn.train_qvalue([data_state, target_qvalue])
                if verbose:
                    print self.nn.loss_qvalue([data_state, target_qvalue])

    def train_action(self, sample_size, check_qvalue=True, verbose=False):
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
                target_qvalue = self.compute_target_qvalue(
                    data_reward, data_state_prime)
                for i in xrange(len(qvalue)):
                    if target_qvalue[i][0] > qvalue[i][0] + self.get_qvalue_knoll():
                        train_state.append(data_state[i])
                        train_action.append(data_action[i])
                data_state = train_state
                data_action = train_action
            if len(data_state) > 0:
                for i in range(self.train_iter):
                    self.nn.train_action([data_state, data_action])
                    if verbose:
                        print self.nn.loss_action([data_state, data_action])
            return len(data_state)
        return 0

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
            target_qvalue = self.compute_target_qvalue(
                data_reward, data_state_prime)
            q = self.nn.loss_qvalue([data_state, target_qvalue])
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


myEnvi = None
myAction = None
myNN = None
myEnviMaster = None
myDeepRL = None
myEye = None

def example_init():
    global myEnvi, myDeepRL, myAction, myNN, myEnviMaster, myEye
    myEnvi = Env(dt, skel_file)
    myAction = Action(myEnvi.skel.controller.actuable_dofs)
    myNN = NN('net_turtle_torque')
    myNN.initialize(
        [len(myEnvi.state()), myAction.dim],
        ckpt_load_dir)

    myEnviMaster = En_Master_Custom(
        max_client,
        gen_env,
        [dt, skel_file],
        Env_Slave_Custom)

    myDeepRL = DeepRL_Multicore(myEnviMaster, myNN, myAction, warmup_file)

    myEye = eye.Eye(render_func=myEnvi.eye_render, setup=True)


def render_callback():
    global gl_init, myEnvi, myDeepRL
    if not gl_init:
        gl_init = True
        example_init()
    if myEnvi is None or myDeepRL is None:
        return
    gl_render.render_ground(color=[0.3, 0.3, 0.3], axis='x')
    gl_render.render_ground(color=[0.3, 0.3, 0.3], axis='y')
    gl_render.render_ground(color=[0.3, 0.3, 0.3], axis='z')
    myEnvi.render()
    if flag['Train']:
        global cnt_target_update
        myDeepRL.run(max_steps)
        if not myDeepRL.is_finished_trainning():
            cnt_target_update += 1
            if cnt_target_update >= max_target_update:
                myDeepRL.save_variables()
                cnt_target_update = 0
                print '------Target Network is updated------',
                print 'avg_reward:', myDeepRL.measure_controller_quality(max_steps)


def keyboard_callback(key):
    if myEnvi is None or myDeepRL is None:
        return
    if key == 'r':
        print 'Key[r]: reset environment'
        global elapsed_time
        myEnvi.reset()
        elapsed_time = 0.0
    elif key == 't':
        print myEnvi.state()
    elif key == 'p':
        elpased = 0.0
        while True:
            myEnvi.step_forward()
            elpased += dt
            if elpased >= 0.03333333:
                print myEnvi.state()
                break
    elif key == '[':
        elpased = 0.0
        while True:
            state = myEnvi.state()
            reward = myEnvi.goodness()
            action = myAction.random(
                [myDeepRL.get_random_noise()] * myAction.dim)
            myEnvi.step(action)
            elpased += dt
            if elpased >= 0.03333333:
                print 'S:', state, 'A:', action, 'R:', reward
                break
    elif key == ' ':
        elapsed = 0.0
        while True:
            state = myEnvi.state()
            action = myDeepRL.get_action(state)
            action = myAction.clamp(action)
            qvalue = myDeepRL.get_qvalue(state)
            reward = myEnvi.goodness()
            myEnvi.step(action)
            elapsed += dt
            elapsed_time += dt
            if elapsed >= 0.01:
                print 't:', elapsed_time, '---------------------------------------'
                print 'S:', state
                print 'A:', action
                print 'R:', reward, 'Q:', qvalue
                myEnvi.goodness(True)
                break
    elif key == 'd':
        print 'Key[d]: train ', not flag['Train']
        flag['Train'] = not flag['Train']
        print 'Train: ', flag['Train']
    elif key == 's':
        print 'Key[s]: save check point file'
        myNN.save(ckpt_save_dir)
    elif key == 'i':
        print 'Key[i]: save eye image'
        myEnvi.eye.update(myEnvi.skel.body('trunk').T)
        myEnvi.eye.save_image('test.png')
        myEnviMaster.reset()
        myEnviMaster.save_eye_image()
    elif key == 'w':
        print 'Key[w]: generate warmup data'
        gen_warmup_data([0.2] * 13, 100, 10)
    elif key == '0':
        sample_idx = myDeepRL.replay_buffer['critic'].sample_idx(10)
        data = myDeepRL.sample('critic', sample_idx)
        print data
    else:
        return False
    return True



pydart.glutgui.glutgui_base.run(
    title='example_turtle',
    trans=[0, 0, -30],
    keyboard_callback=keyboard_callback,
    render_callback=render_callback)
