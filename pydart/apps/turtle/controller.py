import numpy as np
from numpy.linalg import inv
import math
import mmMath
import eye
import action as ac

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
        # self.action_default = [[-1.25567921, 0.6118376, 0.53513041, 0.28105493, 0.78491477, -0.65140349],\
        #     [-1.25567921, 0.6118376, -0.53513041, -0.28105493, -0.78491477, 0.65140349], 1.5]
        self.action_default = [[-1.4597904546361948, 0.43397878147503394, 0.5187277879484047, 0.6483358027166627, 0.38234268646660213, -0.5799882847313336], [-1.4597904546361948, 0.43397878147503394, -0.5187277879484047, -0.6483358027166627, -0.38234268646660213, 0.5799882847313336], 1.5]
        self.action = []
        self.new_wingbeat = True
        self.cnt_wingbeat = 0
        self.time = 0.0
        self.reset()
    def is_new_wingbeat(self):
        return self.new_wingbeat
    def get_num_wingbeat(self):
        return self.cnt_wingbeat
    def reset(self):
        self.tau_sum = 0.0
        del self.action[:]
        self.action.append(self.action_default)
        self.action.append(self.action_default)
        self.new_wingbeat = True
        self.cnt_wingbeat = 0
        self.time = 0.0
    def add_action(self, action, is_delta_moode=False):
        if is_delta_moode:
            self.action.append(ac.add(self.action_default,action))
        else:
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
        bodies = []
        # bodies = ['left_arm', 'right_arm', 'left_hand', 'right_hand']
        # bodies = ['left_arm', 'right_arm']
        for name in bodies:
            body = self.skel.body(name)
            l = body.world_com() - p_trunk
            v = body.world_com_velocity()
            state.append(np.dot(R_trunk_inv,l))
            state.append(np.dot(R_trunk_inv,v))
        return np.array(state).flatten().tolist()
    def get_action_size(self):
        return len(self.action_default[0])+len(self.action_default[1])+1
    def get_state_size(self):
        return len(self.get_state())
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
        # Generate a current target posture
        action_before = self.action[-2]
        action_cur = self.action[-1]
        phase = self.period_to_phase(self.time, action_cur[2])
        alpha = 2.0*phase/math.pi
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

        # Check if wheather new wingbeat 
        self.time += self.skel.world.dt
        if self.time > action_cur[2]:
            self.time = 0.0
            self.new_wingbeat = True
            self.cnt_wingbeat += 1
        else:
            self.new_wingbeat = False

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
