from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import time
import pydart
import math
import numpy as np
import controller
import aerodynamics
import eye
import nn
import deepRL
import scene
from PIL import Image
import profile
import pickle
import json
import action as ac
import optimizer
import mmMath
import myworld
import gl_render

np.set_printoptions(precision=3)

dt = 1.0/600.0
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle.skel'
wall_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/wall.urdf'
warmup_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/warmup/0.15_10000_8.warmup'

print('Example: turtle')

pydart.init()
print('pydart initialization OK')

# Program interactions
state = {}
state['Force'] = np.zeros(3)
state['ImpulseDuration'] = 0
state['DrawAeroForce'] = True
state['DrawGround'] = True
state['DrawJoint'] = True
state['DrawGrid'] = True
state['DrawHistory'] = True
state['EnableAerodynamics'] = True
state['DrawScene'] = True
state['DeepControl'] = False
state['DeepTrainning'] = False
state['DeepPingPongMode'] = True
state['DeepTrainningResultShowMax'] = 5
state['DeepTrainningResultShowCnt'] = 5
state['Test'] = False

aero_force = []
profile = profile.Profile()

grid = []

class History:
    class aHistory:
        def __init__(self, goal):
            self.goal = goal
            self.path = []
            self.state = []
            self.action = []
            self.qvaule = 0
        def add_path(self, transform):
            self.path.append(transform)
    def __init__(self, max_history=10):
        self.history = []
        self.max_history = max_history
    def reset(self, goal=None):
        del self.history[:]
        if goal is not None:
            self.begin(goal)
    def begin(self, goal):
        if len(self.history) >= self.max_history:
            del self.history[0]
        self.history.append(History.aHistory(goal))
    def add_path(self, transform):
        self.history[-1].add_path(transform)
    def set_goal(self, goal):
        self.history[-1].set_goal(goal)

def apply_aerodynamics(skel):
    aero_force = []
    
    for i in range(skel.num_bodies()):
        body = skel.body(i)
        #print body.name

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
            p1 = com + 0.5*d[i]*n1
            p2 = com + 0.5*d[i]*n2
            area = d[(i+1)%3]*d[(i+2)%3]

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
            f = aerodynamics.compute(v,n,a)
            p_local = np.dot(invT,np.append(p,1.0))
            body.add_ext_force_at(f,p_local[0:3])
            aero_force.append([p,f])

    return aero_force

def step_callback(world):
    skel = world.skels[0]
    # if state['EnableAerodynamics']:
    #     apply_aerodynamics(skel)
    
    # if skel.controller.is_new_wingbeat():
    #     print 'action: ', skel.controller.get_action()
    #     print 'state: ', skel.controller.get_state()
    #     print 'velocity: ', skel.body('trunk').world_com_velocity()

    if state['DeepTrainning']:
        if not rl.is_finished_trainning():
            rl.run(10, 10)
            if not rl.is_warming_up():
                state['DeepTrainning'] = False
                state['DeepControl'] = True
                world.reset()
                skel.controller.reset()
                scene.perturbate()
                history.reset(scene.get_pos())
                profile.print_time()
        # if rl.is_finished_trainning():
        #     mynn.save_file()

    if state['DeepControl']:
        if skel.controller.is_new_wingbeat():
            # state_eye = skel.controller.get_eye().get_image(skel.body('trunk').T)
            state_sensor = rl.sensor()
            state_skel = skel.controller.get_state()
            action = rl.get_action(state_sensor,state_skel)
            skel.controller.add_action(action,True)
            qvalue = rl.get_qvalue(state_sensor,state_skel)
            history.add_path(skel.body('trunk').T)
            print 'R:', np.array([scene.score()]),
            print 'Q:', np.array([qvalue]), 
            print '\tS:', np.array(state_sensor), np.array(state_skel), 
            print '\tA:', 
            ac.pprint(action)
            if skel.controller.get_num_wingbeat() >= 30:
                show_cnt = state['DeepTrainningResultShowCnt']
                show_cnt -= 1
                if show_cnt <= 0:
                    show_cnt = state['DeepTrainningResultShowMax']
                    if state['DeepPingPongMode']:
                        if rl.is_finished_trainning():
                            state['DeepTrainning'] = False
                            state['DeepControl'] = True
                        else:
                            state['DeepTrainning'] = True
                            state['DeepControl'] = False
                    else:
                        state['DeepTrainning'] = False
                        state['DeepControl'] = True
                    history.reset()
                state['DeepTrainningResultShowCnt'] = show_cnt
                world.reset()
                skel.controller.reset()
                scene.perturbate()
                history.begin(scene.get_pos())

    if state['Test']:
        if skel.controller.is_new_wingbeat():
            global warmup_data_cnt
            # world.reset()
            t = warmup_data[warmup_data_cnt]
            action = ac.add(skel.controller.get_action_default(), ac.format(t[2]))
            skel.controller.add_action(action)
            warmup_data_cnt = warmup_data_cnt + 1
    
    if False and skel.controller.is_new_wingbeat():
        action_default = skel.controller.get_action_default()
        action_random = ac.random([0.2]*ac.length())
        action = ac.add(action_default, action_random)
        skel.controller.add_action(action)
        print action_random

    scene.update()

    # print 'time: ', world.time()

    # global state
    # if state['ImpulseDuration'] > 0:
    #     f = state['Force']
    #     state['ImpulseDuration'] -= 1
    #     world.skel.body('h_spine').add_ext_force(f)
    # else:
    #     state['Force'] = np.zeros(3)
def render_callback():
    global aero_force

    if state['DrawAeroForce']:
        glColor3d(1.0, 0.0, 0.0)
        glLineWidth(2.0)
        for i in range(len(aero_force)):
            p = aero_force[i][0]
            f = aero_force[i][1]
            e = p+0.001*f
            glPushMatrix()
            glTranslated(p[0],p[1],p[2])
            glutSolidSphere(0.01, 10, 10)
            glPopMatrix()
            glBegin(GL_LINES)
            glVertex3d(p[0],p[1],p[2])
            glVertex3d(e[0],e[1],e[2])
            glEnd()

    if state['DrawGround']:
        gl_render.render_ground()

    if state['DrawJoint']:
        for i in range(skel.num_joints()):
            joint = skel.joint(i)
            # print joint.name
            T = joint.transformation()
            gl_render.render_transform(T,scale=0.5)

    if state['DrawScene']:
        scene.render()

    if state['DrawGrid']:
        for g in grid:
            gl_render.render_transform(g,0.25,0.25)

    if state['DrawHistory']:
        for h in history.history:
            gl_render.render_path(h.path,scale=0.2)

def keyboard_callback(world, key):
    skel = world.skels[0]
    """ Programmable interactions """
    global state
    if key == '1':
        state['Force'][0] = 50
        state['ImpulseDuration'] = 100
        print('push forward')
    elif key == '2':
        state['Force'][0] = -50
        state['ImpulseDuration'] = 100
        print('push backward')
    elif key == '3':
        state['Force'][2] = 50
        state['ImpulseDuration'] = 100
        print('push right')
    elif key == '4':
        state['Force'][2] = -50
        state['ImpulseDuration'] = 100
        print('push left')
    elif key == 's':
        # world.save('test_world.txt')
        mynn.save_file()
    elif key == 'r':
        world.reset()
        skel.controller.reset()
        scene.perturbate()
        history.reset()
        history.begin(scene.get_pos())
        del trajectory[:]
        print scene.score()
    elif key == 't':
        rl.init_step()
        rl.step(0.2, True)
    elif key == 'e':
        state_eye = skel.controller.get_eye().get_image(skel.body('trunk').T)
        state_skel = skel.controller.get_state()
        action = rl.get_action(state_eye,state_skel)#,skel.controller.get_action_default())
        print action
        # skel.controller.add_action(action)
    elif key == 'a':
        size_accum = rl.get_buffer_size_accumulated()
        rl.save_replay_test("save_"+str(size_accum)+".txt")
    elif key == 'd':
        state['DeepTrainning'] = True
        state['DeepTrainningResultShowCnt'] = state['DeepTrainningResultShowMax']
        pydart.glutgui.set_play_speed(10.0)
        pydart.glutgui.play(True)
        profile.begin()
    elif key == 'g':
        print 'reset'
        state['DeepTrainning'] = False
        state['DeepControl'] = True
        state['DeepTrainningResultShowCnt'] = state['DeepTrainningResultShowMax']
        custom_world.reset()
        scene.perturbate()
        history.reset()
        history.begin(scene.get_pos())
        pydart.glutgui.set_play_speed(10.0)
        pydart.glutgui.play(True)
    elif key == 'w':
        print '----- warmup data generation -----'
        gen_warmup_data(0.15, 10000, 8)
    elif key == '9':
        print world.states()
    elif key == '0':
        print('test')
        # tb = pydart.glutgui.Trackball(phi=-1.4, theta=-6.2, zoom=1.0,
        #                         rot=[-0.05, 0.07, -0.01, 1.00],
        #                         trans=[0.02, 0.09, -3.69])
        # pydart.glutgui.set_trackball(tb)
        eye = skel.controller.get_eye()
        eye.update(skel.body('trunk').T)
        eye.save_image('test.png')
        im = eye.get_image()
        print im
    elif key == 'g':
        global grid
        custom_world.reset()
        num_wingbeat = 2
        
        skel = custom_world.skel
        controller = custom_world.skel.controller

        T_0 = skel.body('trunk').T
        T_1 = []
        T_2 = []

        while True:
            custom_world.step()
            if controller.is_new_wingbeat():
                if controller.get_num_wingbeat() == 1:
                    T_1 = skel.body('trunk').T
                elif controller.get_num_wingbeat() == 2:
                    T_2 = skel.body('trunk').T
                if controller.get_num_wingbeat() >= num_wingbeat:
                    break

        grid = gen_grid([T_0,T_1,T_2],[0.15*math.pi,0.15*math.pi,0.75])
    elif key == 'o':
        result = optimizer.run(optimizer.obj_func_straight, optimizer.result_func_straight)
        print result
    else:
        return False
    return True

def gen_grid(transforms,size):
    grid = []

    R_0,p_0 = mmMath.T2Rp(transforms[0])
    R_1,p_1 = mmMath.T2Rp(transforms[1])
    R_2,p_2 = mmMath.T2Rp(transforms[2])

    angle_spread_lr = size[0]
    angle_spread_tb = size[1]
    radius = size[2]*mmMath.length(p_2-p_1)

    v = p_2-p_0
    unit_v = v/mmMath.length(v)

    for i in np.linspace(-angle_spread_lr,angle_spread_lr,5):
        for j in np.linspace(-angle_spread_tb,angle_spread_tb,5):
            for r in np.linspace(-radius,radius,5):
                R = mmMath.getSO3ByEuler([i,j,0.0])
                p = p_0 + np.dot(R,v+r*unit_v)
                grid.append(mmMath.Rp2T(R,p))
    return grid
def gen_warmup_data(file_name, sigma=0.15, num_episode=10000, num_wingbeat=10):
    data = []
    action_default = skel.controller.get_action_default()
    sigma_n = [sigma]*ac.length()
    for ep in xrange(num_episode):
        custom_world.reset()
        for i in xrange(num_wingbeat+1):
            q_skel_init = skel.q
            action_random = ac.random(sigma_n)
            action = ac.add(action_default, action_random)
            skel.controller.add_action(action)
            while True:
                custom_world.step()
                if skel.controller.is_new_wingbeat():
                    break
            q_skel_term = skel.q
            if i != 0:
                data.append([q_skel_init, q_skel_term, action])
        if (ep+1)%1 == 0:
            print (ep+1), 'episode generated'
    f = open(sigma+'_'+num_episode+'_'+num_wingbeat+'.warmup', 'w')
    pickle.dump(data, f)
    f.close()

#
# Initialize world and controller
#

custom_world = myworld.Myworld(dt, skel_file)

world = custom_world.get_world()
skel = custom_world.get_skeleton()
# world.add_skeleton(wall_file)

def gen_scene(stride=3.0, size=1):
    pos = []
    radius = []
    for z in range(size):
        pos.append(np.array([0, 1.0, z*stride+2.0]))
        radius.append(0.5)
    return pos, radius
scene_p, scene_r = gen_scene()
scene = scene.Scene(skel, scene_p, scene_r)
# scene = scene.Scene(skel, [[0.0,1.0,7.0],0.5])

eye = eye.Eye(world=world,scene=scene)
skel.controller = controller.Controller(world, skel, eye)

#
# Initialize nn and RL
#
mynn = nn.MyNNSimple('net')
mynn.initialize([3,\
    skel.controller.get_state_size(),\
    skel.controller.get_action_size()])
# rl = deepRL.DeepRL(world, skel, scene, mynn, "warming_up_db.txt")
# rl = deepRL.DeepRLSimple(world, skel, scene, mynn, "warming_up_db.txt")
rl = deepRL.DeepRLSimple(custom_world, scene, mynn, warmup_file)
# rl = deepRL.DeepRLSimple(custom_world, scene, mynn)

# # Load warmup data for RL
# warmup_data_cnt = 0
# warmup_data_file = 'warming_up_db.txt'
# f = open(warmup_data_file, 'r')
# warmup_data = []
# if state['Test']:
#     warmup_data = pickle.load(f)
#     print "[Warmup data loaded]: ", len(warmup_data)

# Initialize character's dynamical state
# : N number of wingbeat will be performed
#
num_init_wingbeat = 2
while True:
    if skel.controller.get_num_wingbeat() >= num_init_wingbeat:
        world.push()
        world.push()
        break;
    world.step()

history = History()
history.begin(scene.get_pos())

# Run the application
if False:#'qt' in sys.argv:
    tb = pydart.qtgui.Trackball(phi=-1.4, theta=-6.2, zoom=1.0,
                                rot=[-0.05, 0.07, -0.01, 1.00],
                                trans=[0.02, 0.09, -3.69])
    pydart.qtgui.run(title='turtle', simulation=world, trackball=tb,
                     step_callback=step_callback,
                     keyboard_callback=keyboard_callback,
                     render_callback=render_callback)
else:
    # tb = pydart.glutgui.Trackball(phi=-1.4, theta=-6.2, zoom=1.0,
    #                             rot=[-0.05, 0.07, -0.01, 1.00],
    #                             trans=[0.02, 0.09, -3.69])
    pydart.glutgui.glutgui.run(title='turtle', simulation=world, trans=[0, 0, -30],
                       step_callback=step_callback,
                       keyboard_callback=keyboard_callback,
                       render_callback=render_callback)
