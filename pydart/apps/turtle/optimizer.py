import math
import numpy as np
import pydart
import action as ac
import myworld
import cma
import multiprocessing
from multiprocessing import Process, Queue
import mmMath
from numpy.linalg import inv

dt = 1.0/600.0
skel_file = '/home/jungdam/Research/AlphaCon/pydart/apps/turtle/data/skel/turtle.skel'

def obj_func_transform(q, idx, world, action, transform):
	skel = world.get_skeleton()

	a_default = skel.controller.get_action_default()
	a_extra = ac.format(ac.flat(action))
	a = ac.add(a_default, a_extra)

	world.reset()
	skel.controller.reset()
	skel.controller.set_action_all(a)

	um_wingbeat = 2

	while True:
		world.step()
		if skel.controller.get_num_wingbeat() >= num_wingbeat:
			break

	R_des,p_des = mmMath.T2Rp(transform)
	R,p = mmMath.T2Rp(skel.body('trunk').T)

	# difference on orientation
	v0 = 1.0 * mmMath.length(mmMath.log(np.dot(inv(R_des),R)))
	# difference on position
	v1 = 1.0 * mmMath.length(p_des-p)
	# penalty on large deviation
	v2 = 10 * np.dot(np.array(a_l),np.array(a_l))
	# penalty on large torque
	v3 = 0.00002 * skel.controller.get_tau_sum() / (num_wingbeat*a[2])
	val = v0 + v1 + v2 + v3
	print '\t', val, v0, v1, v2, v3

	q.put([idx, val])

def obj_func_straight(q, idx, world, action):
    skel = world.get_skeleton()

    a_default = skel.controller.get_action_default()
    a_l = action[0:6].tolist()
    a_r = ac.mirror(a_l)
    a_t = 0.0
    
    a_extra = [a_l,a_r,a_t]
    a = ac.add(a_default, a_extra)

    world.reset()
    skel.controller.reset()
    skel.controller.set_action_all(a)

    num_wingbeat = 8

    t0 = skel.body('trunk').world_com()

    while True:
        world.step()
        if skel.controller.get_num_wingbeat() >= num_wingbeat:
            break

    t1 = skel.body('trunk').world_com()

    diff = t1-t0
    v0 = 10.0 * (diff[0]*diff[0] + diff[1]*diff[1])
    v1 = 30 * math.exp(-0.01*diff[2]*diff[2])
    v2 = 100 * np.dot(np.array(a_l),np.array(a_l))
    v3 = 0.0002 * skel.controller.get_tau_sum() / (num_wingbeat*a[2])
    val = v0 + v1 + v2 + v3
    print '\t', val, v0, v1, v2, v3
    
    q.put([idx, val])

def result_func_straight(result,controller):
	a_default = controller.get_action_default()
	a_l = result[0].tolist()
	a_r = ac.mirror(a_l)
	a_t = 0.0

	a_extra = [a_l,a_r,a_t]
	return ac.add(a_default, a_extra)

def run(obj_func, result_func, options=None):
	print('-----------Start Optimization-----------')
	num_cores = multiprocessing.cpu_count()
	num_pop = max(8, 1*num_cores)
	num_gen = 100

	myWorlds = []
	for i in range(num_pop):
		myWorlds.append(myworld.Myworld(dt, skel_file))

	opts = cma.CMAOptions()
	opts.set('pop', num_pop)
	es = cma.CMAEvolutionStrategy(6*[0.0], 0.2, opts)
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
	return result_func(es.result(),myWorlds[0].skel.controller)