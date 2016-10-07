from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pydart
import controller
import numpy as np
import pickle
import action as ac
from numpy.linalg import inv
import mmMath

state = {}
state['DrawAxis'] = True
state['DrawGrid'] = True
state['DrawDataAbs'] = True
state['DrawDataRelGlobal'] = True
state['DrawDataRelLocal'] = True

data_abs = None
data_rel_global = None
data_rel_local = None

dt = 1.0/600.0
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle.skel'
data_file = './data/warmup/0.2_10_10_torque.warmup'
pydart.init()

world = pydart.create_world(dt, skel_file)
skel = world.skels[0]
skel.controller = controller.Controller(world, skel)

def load_data(file_name, world, skel):

    f = open(file_name, 'r')
    data = pickle.load(f)
    size = len(data)
    data_abs = []
    data_rel_local = []
    data_rel_global = []

    print len(data), 'data loaded'

    cnt = 0
    for d in data:
        state_skel_init = d[0]
        state_skel_term = d[1]
        action = d[2]

        # print action

        #
        world.reset()
        skel.controller.reset()
        skel.set_states(state_skel_init)
        # world.step(False,False)

        R_1,p_1 = mmMath.T2Rp(skel.body('trunk').T)
        # p_1 = skel.body('trunk').world_com()

        # skel.controller.add_action(action)
        # while True:
        # 	world.step()
        # 	if skel.controller.is_new_wingbeat():
        # 		break

        #
        world.reset()
        skel.controller.reset()
        skel.set_states(state_skel_term)
        # world.step(False,False)

        R_2,p_2 = mmMath.T2Rp(skel.body('trunk').T)
        # p_2 = skel.body('trunk').world_com()

        data_rel_global.append(p_2-p_1)
        data_rel_local.append(np.dot(inv(R_1),p_2-p_1))
        data_abs.append(p_1)

        cnt += 1
        if cnt%1000==0:
            print cnt, "data processed"

    return data_abs,data_rel_global,data_rel_local

def render_callback():
    global state
    global data_rel_global
    global data_rel_local
    if state['DrawAxis']:
        glLineWidth(5.0)
        glBegin(GL_LINES)
        glColor3d(1,0,0)
        glVertex3d(0,0,0)
        glVertex3d(1,0,0)
        glColor3d(0,1,0)
        glVertex3d(0,0,0)
        glVertex3d(0,1,0)
        glColor3d(0,0,1)
        glVertex3d(0,0,0)
        glVertex3d(0,0,1)
        glEnd()
    if state['DrawGrid']:
        l = 2.0
        dl = 0.1
        n = int(l/dl)

        glColor3d(0.5, 0.5, 0.5)
        glLineWidth(0.5)

        # xz plane
        for i in range(2*n+1):
            glBegin(GL_LINES)
            glVertex3d(-l+i*dl,0,-l)
            glVertex3d(-l+i*dl,0,l)
            glEnd()
        for i in range(2*n+1):
            glBegin(GL_LINES)
            glVertex3d(-l,0,-l+i*dl)
            glVertex3d(l,0,-l+i*dl)
            glEnd()
        # xy plane
        for i in range(2*n+1):
            glBegin(GL_LINES)
            glVertex3d(-l+i*dl,-l,0)
            glVertex3d(-l+i*dl,l,0)
            glEnd()
        for i in range(2*n+1):
            glBegin(GL_LINES)
            glVertex3d(-l,-l+i*dl,0)
            glVertex3d(l,-l+i*dl,0)
            glEnd()
        # yz plane
        for i in range(2*n+1):
            glBegin(GL_LINES)
            glVertex3d(0,-l+i*dl,-l)
            glVertex3d(0,-l+i*dl,l)
            glEnd()
        for i in range(2*n+1):
            glBegin(GL_LINES)
            glVertex3d(0,-l,-l+i*dl)
            glVertex3d(0,l,-l+i*dl)
            glEnd()
    if state['DrawDataRelGlobal']:
        glColor3d(1.0, 1.0, 0.0)
        glBegin(GL_POINTS)
        for d in data_rel_global:
            glVertex3d(d[0],d[1],d[2])
        glEnd()
    if state['DrawDataRelLocal']:
        glColor3d(0.0, 1.0, 1.0)
        glBegin(GL_POINTS)
        for d in data_rel_local:
            glVertex3d(d[0],d[1],d[2])
        glEnd()
    if state['DrawDataAbs']:
        glColor3d(1.0, 0.0, 1.0)
        glBegin(GL_POINTS)
        for d in data_abs:
            glVertex3d(d[0],d[1],d[2])
        glEnd()

def keyboard_callback(key):
    """ Programmable interactions """
    global state
    if key == '1':
        state['DrawDataAbs'] = not state['DrawDataAbs']
    elif key == '2':
        state['DrawDataRelGlobal'] = not state['DrawDataRelGlobal']
    elif key == '3':
        state['DrawDataRelLocal'] = not state['DrawDataRelLocal']
    else:
        return False
    return True

data_abs,data_rel_global,data_rel_local = load_data(data_file, world, skel)

pydart.glutgui.glutgui_base.run(
    title='Data Checker', 
    trans=[0, 0, -30],
    keyboard_callback=keyboard_callback,
    render_callback=render_callback)