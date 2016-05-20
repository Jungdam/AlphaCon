import numpy as np
from numpy.linalg import inv
import math
import mmMath
import eye

class Controller:
    """ Add damping force to the skeleton """
    def __init__(self, world, skel, eye=None):
        self.world = world
        self.skel = skel
        self.eye = eye
        self.h = world.dt
        ndofs = self.skel.ndofs
        self.qhat_init = self.skel.q
        self.qhat = self.skel.q
        self.Kp = np.diagflat([0.0] * 6 + [5000.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [500.0] * (ndofs - 6))
        self.tau_sum = 0.0
        # self.action_base = [[-1.5, 0.5, 0.5, 0.0, 1.0, -0.5], [-1.5, 0.5, -0.5, 0.0, -1.0, 0.5], 1.5]
        # self.action_default = [[0.24432079, 0.1118376, 0.03513041, 0.28105493, -0.21508523, -0.15140349], \
        #     [0.24432079, 0.1118376, -0.03513041, -0.28105493, 0.21508523, 0.15140349], 1.5]
        self.action_default = [[-1.25567921, 0.6118376, 0.53513041, 0.28105493, 0.78491477, -0.65140349],\
            [-1.25567921, 0.6118376, -0.53513041, -0.28105493, -0.78491477, 0.65140349], 1.5]
        self.action = []
        self.new_wingbeat = True
        self.cnt_wingbeat = 0
        self.time = 0.0
        self.reset()
    def is_new_wingbeat(self):
        return self.new_wingbeat
    def reset(self):
        self.tau_sum = 0.0
        del self.action[:]
        self.action.append(self.action_default)
        self.action.append(self.action_default)
        self.new_wingbeat = True
        self.cnt_wingbeat = 0
        self.time = 0.0
        # # self.set_action_all(p)
        # # d_l = np.array([-0.05552078,  0.17455198,  0.01879346,  0.08439687,  0.09147024, -0.16267498])
        # # # d_t = 0.02435852
        # # # d_l = np.array([ 0.19317016,  0.09538803,  0.15941522,  0.27894028, -0.11860695, -0.06225413])
        # # d_l = np.array([ 0.529366  , -0.40631411, -0.56509792,  0.53919454, -0.15038238, -1.45177158])
        # # d_l = np.array([-0.03953661, -0.08070696, -0.01347923,  0.01058312,  0.09924998, 0.11067133])
        # # d_l = np.array([ 0.12364231,  0.12258705, -0.12238833,  0.11594259, -0.14220551, -0.07678395])
        # # # d_l = np.array([-0.22361199, -0.24188249, -0.11789932,  0.25582372, -0.1583279 , 0.10787422])
        # # d_l = np.array([-0.04504375, -0.14538448,  0.09693785,  0.00043086, -0.18594123, 0.01955406])
        # d_l = np.array([-0.2779736 , -0.06742797, -0.04015606,  0.23646649, -0.14967497, -0.15978058])
        # d_l = np.array([ 0.24432079,  0.1118376 ,  0.03513041,  0.28105493, -0.21508523, -0.15140349])
        # d_r = np.array([d_l[0],d_l[1],-d_l[2],-d_l[3],-d_l[4],-d_l[5]])
        # d_t = 0.0        
        # l = np.array(self.action_default[0]) + d_l
        # r = np.array(self.action_default[1]) + d_r
        # t = self.action_default[2] + d_t
        # self.set_action_all([l.tolist(),r.tolist(),t])
    def add_action(self, action):
        self.action.append(action)
    def set_action(self, action):
        self.action = action
    def set_action_all(self, action):
        for i in range(len(self.action)):
            self.action[i] = action
    def get_action(self):
        return self.action[-1]
    def get_action_all(self):
        return self.action
    def get_action_default(self):
        return self.action_default
    def get_tau_sum(self):
        return self.tau_sum
    def get_state(self):
        # state
        state = []
        # prepare for computing local coordinate of the trunk
        body_trunk = self.skel.body('trunk')
        R_trunk,p_trunk = mmMath.T2Rp(body_trunk.T)
        R_trunk_inv = inv(R_trunk)
        # trunk state 
        vel_trunk = body_trunk.world_com_velocity()
        state.append(np.dot(R_trunk_inv,vel_trunk))
        # other bodies state
        bodies = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
        for name in bodies:
            body = self.skel.body(name)
            l = body.world_com() - p_trunk
            state.append(np.dot(R_trunk_inv,l))
        return np.array(state).flatten().tolist()
    def get_eye(self):
        return self.eye
    def reset_tau_sum(self):
        self.tau_sum = 0.0
    def period_to_phase(self, e_t, period):
        if e_t <= 0.0 or period <= 0.0:
            return 0.0
        else:
            return (2.0*math.pi) * (1.0-(period-e_t)/period)
    def move_func(self, axis, u_up, u_down, phase):
        theta = phase
        value = 0.0

        if axis == 'sweep' or axis == 'twist':
            if phase >= 0.0 and phase <= math.pi:
                value = u_down
            else:
                value = 0.5 * (u_up-u_down) * (1-math.cos(2.0*theta)) + u_down
        elif axis == 'dihedral':
            value = 0.5 * (u_up-u_down) * (1+math.cos(theta)) + u_down

        return value
    def update_target_pose(self):
        # Check if wheather new wingbeat 
        self.time += self.skel.world.dt
        if self.time > self.action[-1][2]:
            self.time = 0.0
            self.new_wingbeat = True
            self.cnt_wingbeat += 1
            if self.eye is not None:
                self.eye.update(self.skel.body('trunk').T)
                # self.eye.save_image('image'+str(self.cnt_wingbeat)+'.png')
            if False and self.cnt_wingbeat > 2:
                noiseL = np.random.uniform(-0.5, 0.5, size=len(self.action_default[0]))
                noiseR = np.random.uniform(-0.5, 0.5, size=len(self.action_default[1]))
                noiseT = np.random.uniform(-0.5, 0.5, size=1)
                l = np.array(self.action_default[0]) + noiseL
                r = np.array(self.action_default[1]) + noiseR
                t = np.array(self.action_default[2]) + noiseT
                action = [l.tolist(),r.tolist(),t[0]]
                self.add_action(action)
        else:
            self.new_wingbeat = False

        # Generate a current target posture
        action_before = self.action[-2]
        action_cur = self.action[-1]
        phase = self.period_to_phase(self.time, action_cur[2])
        alpha = phase/math.pi
        if alpha > 1.0:
            alpha = 1.0
        for i in range(2):
            a1 = self.move_func('twist', action_before[i][0], action_before[i][1], phase)
            a2 = self.move_func('sweep', action_before[i][2], action_before[i][3], phase)
            a3 = self.move_func('dihedral', action_before[i][4], action_before[i][5], phase)
            b1 = self.move_func('twist', action_cur[i][0], action_cur[i][1], phase)
            b2 = self.move_func('sweep', action_cur[i][2], action_cur[i][3], phase)
            b3 = self.move_func('dihedral', action_cur[i][4], action_cur[i][5], phase)
            c1 = (1.0-alpha)*a1 + alpha*b1
            c2 = (1.0-alpha)*a2 + alpha*b2
            c3 = (1.0-alpha)*a3 + alpha*b3
            if i == 0:
                self.qhat["j_arm_left_x"] = self.qhat_init["j_arm_left_x"] + c1
                self.qhat["j_arm_left_y"] = self.qhat_init["j_arm_left_y"] + c2
                self.qhat["j_arm_left_z"] = self.qhat_init["j_arm_left_z"] + c3
            else:
                self.qhat["j_arm_right_x"] = self.qhat_init["j_arm_right_x"] + c1
                self.qhat["j_arm_right_y"] = self.qhat_init["j_arm_right_y"] + c2
                self.qhat["j_arm_right_z"] = self.qhat_init["j_arm_right_z"] + c3
    def compute(self):

        self.update_target_pose()

        skel = self.skel

        if skel.world.t == 0.0:
            skel.set_positions(self.qhat)

        # apply SPD (Stable PD Control - Tan et al.)
        invM = inv(skel.M + self.Kd * self.h)
        p = -self.Kp.dot(skel.q + skel.qdot * self.h - self.qhat)
        d = -self.Kd.dot(skel.qdot)
        qddot = invM.dot(-skel.c + p + d + skel.constraint_forces())
        tau = p + d - self.Kd.dot(qddot) * self.h
        # tau = p
        # Make sure the first six are zero
        tau[0:6] = 0.0
        for i in range(len(tau)):
            if tau[i] > 1000.0:
                tau[i] = 1000.0
            if tau[i] < -1000.0:
                tau[i] = -1000.0
        self.tau_sum += np.linalg.norm(tau)
        return tau
