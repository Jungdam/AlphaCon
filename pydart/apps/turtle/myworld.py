import math
import pydart
import numpy as np
import controller
import aerodynamics


def apply_aerodynamics(skel):
    for i in range(skel.num_bodies()):
        body = skel.body(i)

        v = body.world_com_velocity()
        d = body.bounding_box_dims()
        T = body.transformation()
        invT = np.linalg.inv(T)
        com = T[0:3, 3]

        positions = []
        normals = []
        areas = []

        for i in range(3):
            n1 = T[0:3, i]
            n2 = -n1
            p1 = com + 0.5 * d[i] * n1
            p2 = com + 0.5 * d[i] * n2
            area = d[(i + 1) % 3] * d[(i + 2) % 3]

            positions.append(p1)
            positions.append(p2)
            normals.append(n1)
            normals.append(n2)
            areas.append(area)
            areas.append(area)

        for i in range(len(positions)):
            p = positions[i]
            n = normals[i]
            a = areas[i]
            f = aerodynamics.compute(v, n, a)
            p_local = np.dot(invT, np.append(p, 1.0))
            body.add_ext_force_at(f, p_local[0:3])


class Myworld:
    def __init__(self, dt, skel_file):
        self.world = pydart.create_world(dt, skel_file)
        self.skel = self.world.skels[0]
        self.skel.controller = controller.Controller(self.world, self.skel)

    def reset(self):
        self.world.reset()
        self.skel.controller.reset()

    def step(self, apply_controller=True, apply_aero=True):
        self.world.step(apply_controller, apply_aero)

    def get_world(self):
        return self.world

    def get_skeleton(self):
        return self.skel

    def get_scene(self):
        return self.scene

    def get_time_step(self):
        return self.world.dt
