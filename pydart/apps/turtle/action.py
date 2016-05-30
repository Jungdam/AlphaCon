import numpy as np

dim_l = 6
dim_r = 6

def flatten(l):
	return list(_flatten_(l))

def _flatten_(*args):
    for x in args:
        if hasattr(x, '__iter__'):
            for y in _flatten_(*x):
                yield y
        else:
            yield x

def add(a, b):
	return list([
		(np.array(a[0])+np.array(b[0])).tolist(),
		(np.array(a[1])+np.array(b[1])).tolist(),
		a[2]+b[2]])
def sub(a, b):
	return list([
		(np.array(a[0])-np.array(b[0])).tolist(),
		(np.array(a[1])-np.array(b[1])).tolist(),
		a[2]-b[2]])
def random(sigma, action=None):
	d = len(sigma)
	a = [0.0]*d
	for i in xrange(d):
		a[i] += np.random.uniform(-sigma[i], sigma[i])
	r = [a[0:dim_l],a[dim_l:dim_r+dim_l],a[dim_r+dim_l]]
	if action is None:
		return r
	else:
		return add(r, a)
def zero():
	return [[0.0]*dim_l,[0.0]*dim_r,0.0]
def length():
	return dim_l+dim_r+1
def flat(a):
	return flatten(a)
def format(a):
	return [a[0:dim_l],a[dim_l:dim_r+dim_l],a[dim_r+dim_l]]