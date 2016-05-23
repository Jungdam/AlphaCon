import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import mmMath

# class Object:
# 	def __init__(self):
# 		self.T = mmMath.I_SE3()
# 	def collide(self, other)

class Map:
	def __init__(self):
		self.pos = []
		self.radius = []
	def collide(self, transform, radius):
		num_obj = len(self.pos)
		for i in range(num_obj):
			p = self.pos[i]
			r = self.radius[i]

			transform[0:3,:]
