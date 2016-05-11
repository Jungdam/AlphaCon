import numpy as np
from numpy.linalg import inv
import math
import mmMath

time = 0.0

class Controller:
    """ Add damping force to the skeleton """
    def __init__(self, skel, h):
        self.h = h
        self.skel = skel
        ndofs = self.skel.ndofs
        self.qhat_init = self.skel.q
        self.qhat = self.skel.q
        self.Kp = np.diagflat([0.0] * 6 + [5000.0] * (ndofs - 6))
        self.Kd = np.diagflat([0.0] * 6 + [500.0] * (ndofs - 6))
        self.preoffset = 0.0
        self.tau_sum = 0.0
        self.param_default = [[-1.5, 0.5, 0.5, 0.0, 1.0, -0.5], [-1.5, 0.5, -0.5, 0.0, -1.0, 0.5], 1.5]
        self.param = [[[-1.5, 0.5, 0.5, 0.0, 1.0, -0.5], [-1.5, 0.5, -0.5, 0.0, -1.0, 0.5], 1.5],
            [[-1.5, 0.5, 0.5, 0.0, 1.0, -0.5], [-1.5, 0.5, -0.5, 0.0, -1.0, 0.5], 1.5]]
        self.new_wingbeat = True

        # self.set_param_all(p)
        # d_l = np.array([-0.05552078,  0.17455198,  0.01879346,  0.08439687,  0.09147024, -0.16267498])
        # # d_t = 0.02435852
        # # d_l = np.array([ 0.19317016,  0.09538803,  0.15941522,  0.27894028, -0.11860695, -0.06225413])
        # d_l = np.array([ 0.529366  , -0.40631411, -0.56509792,  0.53919454, -0.15038238, -1.45177158])
        # d_l = np.array([-0.03953661, -0.08070696, -0.01347923,  0.01058312,  0.09924998, 0.11067133])
        # d_l = np.array([ 0.12364231,  0.12258705, -0.12238833,  0.11594259, -0.14220551, -0.07678395])
        # # d_l = np.array([-0.22361199, -0.24188249, -0.11789932,  0.25582372, -0.1583279 , 0.10787422])
        # d_l = np.array([-0.04504375, -0.14538448,  0.09693785,  0.00043086, -0.18594123, 0.01955406])
        d_l = np.array([-0.2779736 , -0.06742797, -0.04015606,  0.23646649, -0.14967497, -0.15978058])
        d_l = np.array([ 0.24432079,  0.1118376 ,  0.03513041,  0.28105493, -0.21508523, -0.15140349])
        d_r = np.array([d_l[0],d_l[1],-d_l[2],-d_l[3],-d_l[4],-d_l[5]])
        d_t = 0.0        
        l = np.array(self.param_default[0]) + d_l
        r = np.array(self.param_default[1]) + d_r
        t = self.param_default[2] + d_t
        self.set_param_all([l.tolist(),r.tolist(),t])
    def is_new_wingbeat(self):
        return self.new_wingbeat
    def reset(self):
        self.set_param_all(self.param_default)
    def add_param(self, param):
        self.param.append(param)
    def set_param(self, param):
        self.param = param
    def set_param_all(self, param):
        for i in range(len(self.param)):
            self.param[i] = param
    def get_param(self):
        return self.param[-1]
    def get_param_all(self):
        return self.param
    def get_param_default(self):
        return self.param_default
    def get_tau_sum(self):
        return self.tau_sum
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
        global time

        param_before = self.param[-2]
        param_cur = self.param[-1]

        time += self.skel.world.dt
        if time > param_cur[2]:
            time = 0.0
            self.new_wingbeat = True
        else:
            self.new_wingbeat = False

        phase = self.period_to_phase(time, param_cur[2])
        alpha = phase/math.pi

        # print phase

        if alpha > 1.0:
            alpha = 1.0

        for i in range(2):

            a1 = self.move_func('twist', param_before[i][0], param_before[i][1], phase)
            a2 = self.move_func('sweep', param_before[i][2], param_before[i][3], phase)
            a3 = self.move_func('dihedral', param_before[i][4], param_before[i][5], phase)

            b1 = self.move_func('twist', param_cur[i][0], param_cur[i][1], phase)
            b2 = self.move_func('sweep', param_cur[i][2], param_cur[i][3], phase)
            b3 = self.move_func('dihedral', param_cur[i][4], param_cur[i][5], phase)

            c1 = (1.0-alpha)*a1 + alpha*b1
            c2 = (1.0-alpha)*a2 + alpha*b2
            c3 = (1.0-alpha)*a3 + alpha*b3

            # print c1, c2, c3

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

        # Check the balance
        COP = skel.body('h_heel_left').to_world([0.05, 0, 0])
        offset = skel.C[0] - COP[0]
        preoffset = self.preoffset
        # print ("offset = %f, preoffset = %f" % (offset, preoffset))

        # Adjust the target pose -- translated from bipedStand app of DART
        foot = skel.dof_indices(["j_heel_left_1", "j_toe_left",
                                 "j_heel_right_1", "j_toe_right"])
        if 0.0 < offset < 0.1:
            k1, k2, kd = 200.0, 100.0, 10.0
            k = np.array([-k1, -k2, -k1, -k2])
            tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
            self.preoffset = offset
        elif -0.2 < offset < -0.05:
            k1, k2, kd = 2000.0, 100.0, 100.0
            k = np.array([-k1, -k2, -k1, -k2])
            tau[foot] += k * offset + kd * (preoffset - offset) * np.ones(4)
            self.preoffset = offset

        # Make sure the first six are zero
        tau[:6] = 0
        return tau
