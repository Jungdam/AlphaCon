from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import sys
import time
import pydart
import math
import numpy as np
import controller
import cma
import aerodynamics
import myworld
import multiprocessing
from multiprocessing import Process, Queue

dt = 1.0/600.0
skel_file = '/home/jungdam/Research/pydart/apps/turtle/data/skel/turtle.skel'

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
state['EnableAerodynamics'] = True

aero_force = []

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

def obj_func(q, idx, world, param):
    
    skel = world.get_skeleton()

    # convert 
    p_default = skel.controller.get_param_default()

    p_l = param[0:6]
    p_r = np.array([p_l[0],p_l[1],-p_l[2],-p_l[3],-p_l[4],-p_l[5]])
    p_t = 0.0 #param[-1]
    p = [(np.array(p_default[0])+p_l).tolist(),(np.array(p_default[1])+p_r).tolist(),(p_default[2]+p_t)]

    world.reset()
    skel.controller.set_param_all(p)
    skel.controller.reset_tau_sum()

    t0 = skel.body('trunk').world_com()

    time_to_simulate = int(5 * p[2])
    num_frame = int(time_to_simulate / world.get_time_step())
    for i in range(num_frame):
        world.step()

    t1 = skel.body('trunk').world_com()

    diff = t1-t0
    v0 = 10.0 * (diff[0]*diff[0] + diff[1]*diff[1])
    v1 = 40 * math.exp(-0.01*diff[2]*diff[2])
    v2 = 10 * np.dot(p_l,p_l)
    v3 = 0.00002 * skel.controller.get_tau_sum() / time_to_simulate
    print '\t', p, v0, v1, v2, v3
    val = v0 + v1 + v2 + v3

    q.put([idx, val])

def step_callback(world):

    if state['EnableAerodynamics']:
        apply_aerodynamics(world.skels[0])

    if world.skels[0].controller.is_new_wingbeat():
        print world.skels[0].body('trunk').world_com_velocity()

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
        lx = 20.0
        lz = 20.0
        nx = 10
        nz = 10
        dx = 2.0*lx/nx
        dz = 2.0*lz/nz

        glColor3d(0.0, 0.5, 0.0)
        glLineWidth(2.0)
        for i in range(nx+1):
            glBegin(GL_LINES)
            glVertex3d(-lx+i*dx,0,-lz)
            glVertex3d(-lx+i*dx,0,lz)
            glEnd()
        for i in range(nz+1):
            glBegin(GL_LINES)
            glVertex3d(-lx,0,-lz+i*dz)
            glVertex3d(lx,0,-lz+i*dz)
            glEnd()

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

    if state['DrawJoint']:
        
        glLineWidth(2.0)
        for i in range(skel.num_joints()):
            joint = skel.joint(i)
            # print joint.name

            T = joint.transformation()
            R = T[0:3, 0:3]
            t = T[0:3, 3]

            glPushMatrix()
            glTranslated(t[0],t[1],t[2])

            glColor3d(1.0, 1.0, 0.0)
            glutSolidSphere(0.05, 10, 10)
            glBegin(GL_LINES)
            
            glColor3d(1,0,0)
            glVertex3d(0,0,0)
            glVertex3d(R[0,0],R[1,0],R[2,0])
            
            glColor3d(0,1,0)
            glVertex3d(0,0,0)
            glVertex3d(R[0,1],R[1,1],R[2,1])

            glColor3d(0,0,1)
            glVertex3d(0,0,0)
            glVertex3d(R[0,2],R[1,2],R[2,2])
            glEnd()
            glPopMatrix()

def keyboard_callback(world, key):
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
        print('save world')
        world.save('test_world.txt')
    elif key == 'o':
        print('-----------Start Optimization-----------')
        num_cores = multiprocessing.cpu_count()
        num_pop = max(8, 1*num_cores)
        num_gen = 50
        
        myWorlds = []
        for i in range(num_pop):
            myWorlds.append(myworld.Myworld(dt, skel_file, step_callback))

        opts = cma.CMAOptions()
        opts.set('pop', num_pop)
        es = cma.CMAEvolutionStrategy(6*[0], 0.2, opts)
        for i in range(num_gen):
            X = es.ask()
            fit = num_pop*[0.0]
            q = Queue()
            ps = []
            for j in range(num_pop):
                p = Process(target=obj_func, args=(q, j, myWorlds[j],X[j]))
                p.start()
                ps.append(p)
            for j in range(num_pop):
                ps[j].join()
            for j in range(num_pop):
                val = q.get()
                fit[val[0]] = val[1]
                print '[', i, val[0], ']', val[1]
            es.tell(X,fit)
        cma.pprint(es.result())

        # es.optimize(obj_func, verb_disp=1)
        # cma.pprint(es.result())
        print('-----------End Optimization-------------')

world = pydart.create_world(dt, skel_file)
skel = world.skels[0]
skel.controller = controller.Controller(skel, dt)

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
    pydart.glutgui.run(title='turtle', simulation=world, trans=[0, 0, -5],
                       step_callback=step_callback,
                       keyboard_callback=keyboard_callback,
                       render_callback=render_callback)