import numpy as np
import basics

default = np.array([\
	-1.25567921, 0.6118376, 0.53513041, 0.28105493, 0.78491477, -0.65140349, \
	-1.25567921, 0.6118376, -0.53513041, -0.28105493, -0.78491477, 0.65140349, \
	1.5])

dim_left = 6
dim_right = 6
dim_others = 1

def get_left(a):
	return a[0:dim_left]
def get_right(a):
	return a[dim_left:dim_left+dim_right]
def mirror(a):
	return np.array([a[0],a[1],-a[2],-a[3],-a[4],-a[5]])
def dim():
	return dim_left+dim_right+dim_others
def length():
	return dim_left+dim_right+dim_others
def zero():
	np.zeros(dim())
def pprint(a):
	print np.array(a[0]), np.array(a[1]), np.array([a[2]])
def random(mu, sigma):
	return np.random.normal(mu, sigma)