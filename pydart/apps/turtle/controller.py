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
        self.action.append(ac.default)
        self.action.append(ac.default)
        self.new_wingbeat = True
        self.cnt_wingbeat = 0
        self.time = 0.0
    def add_action(self, action, delta_moode=False):
        if delta_moode:
            self.action.append(ac.default+action)
        else:
            self.action.append(action)
        if self.action[-1][-1] <= 0.0:
            print action
            raise Exception('[Action]', 'negative time length')
    def get_tau_sum(self):
        return self.tau_sum
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
    def compute_target_pose_oneside(self, action_before, action_cur, phase):
        alpha = min(2.0*phase/math.pi, 1.0)
        a1 = self.move_func('twist', action_before[0], action_before[1], phase)
        a2 = self.move_func('sweep', action_before[2], action_before[3], phase)
        a3 = self.move_func('dihedral', action_before[4], action_before[5], phase)
        b1 = self.move_func('twist', action_cur[0], action_cur[1], phase)
        b2 = self.move_func('sweep', action_cur[2], action_cur[3], phase)
        b3 = self.move_func('dihedral', action_cur[4], action_cur[5], phase)
        c1 = (1.0-alpha)*a1 + alpha*b1
        c2 = (1.0-alpha)*a2 + alpha*b2
        c3 = (1.0-alpha)*a3 + alpha*b3
        return c1, c2, c3
    def update_target_pose(self):
        # Generate a current target posture
        action_before = self.action[-2]
        action_cur = self.action[-1]
        time_cur = action_cur[-1]
        phase = self.period_to_phase(self.time, time_cur)
        
        dqx, dqy, dqz = self.compute_target_pose_oneside(
            ac.get_left(action_before), ac.get_left(action_cur), phase)
        self.qhat["j_arm_left_x"] = self.qhat_init["j_arm_left_x"] + dqx
        self.qhat["j_arm_left_y"] = self.qhat_init["j_arm_left_y"] + dqy
        self.qhat["j_arm_left_z"] = self.qhat_init["j_arm_left_z"] + dqz

        dqx, dqy, dqz = self.compute_target_pose_oneside(
            ac.get_right(action_before), ac.get_right(action_cur), phase)
        self.qhat["j_arm_right_x"] = self.qhat_init["j_arm_right_x"] + dqx
        self.qhat["j_arm_right_y"] = self.qhat_init["j_arm_right_y"] + dqy
        self.qhat["j_arm_right_z"] = self.qhat_init["j_arm_right_z"] + dqz

        # for i in range(2):
        #     a1 = self.move_func('twist', action_before[i][0], action_before[i][1], phase)
        #     a2 = self.move_func('sweep', action_before[i][2], action_before[i][3], phase)
        #     a3 = self.move_func('dihedral', action_before[i][4], action_before[i][5], phase)
        #     b1 = self.move_func('twist', action_cur[i][0], action_cur[i][1], phase)
        #     b2 = self.move_func('sweep', action_cur[i][2], action_cur[i][3], phase)
        #     b3 = self.move_func('dihedral', action_cur[i][4], action_cur[i][5], phase)
        #     c1 = (1.0-alpha)*a1 + alpha*b1
        #     c2 = (1.0-alpha)*a2 + alpha*b2
        #     c3 = (1.0-alpha)*a3 + alpha*b3
        #     if i == 0:
        #         self.qhat["j_arm_left_x"] = self.qhat_init["j_arm_left_x"] + c1
        #         self.qhat["j_arm_left_y"] = self.qhat_init["j_arm_left_y"] + c2
        #         self.qhat["j_arm_left_z"] = self.qhat_init["j_arm_left_z"] + c3
        #     else:
        #         self.qhat["j_arm_right_x"] = self.qhat_init["j_arm_right_x"] + c1
        #         self.qhat["j_arm_right_y"] = self.qhat_init["j_arm_right_y"] + c2
        #         self.qhat["j_arm_right_z"] = self.qhat_init["j_arm_right_z"] + c3

        # Check if wheather new wingbeat 
        self.time += self.skel.world.dt
        if self.time > time_cur:
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
