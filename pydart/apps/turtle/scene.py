import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import mmMath

# class Object:
# 	def __init__(self):
# 		self.T = mmMath.I_SE3()
# 	def collide(self, other)

class Scene:
	def __init__(self, pos=None, radius=None):
		self.pos_init = []
		self.radius_init = []
		self.pos = []
		self.radius = []
		self.hit = []
		self.size = 0
		if pos != None and radius != None:
			self.generate(pos, radius)
	def score(self):
		cnt = 0
		for i in range(self.size):
			if self.hit[i]:
				cnt += 1
		return float(cnt)
	def update(self, transform, radius):
		for i in range(self.size):
			p = self.pos[i]
			r = self.radius[i]+radius
			l = np.linalg.norm(p-transform[0:3,3])
			if l < r:
				self.hit[i] = True
	def generate(self, pos, radius):
		self.pos_init = pos[:]
		self.radius_init = radius[:]
		self.pos = pos[:]
		self.radius = radius[:]
		self.size = len(pos)
		self.hit = [False] * self.size
	def perturbate(self, sigma_pos=[0.5, 0.5, 0.5], sigma_radius=0.01):
		for i in range(self.size):
			noise_pos = np.array([
				np.random.normal(0.0, sigma_pos[0]),
				np.random.normal(0.0, sigma_pos[1]),
				np.random.normal(0.0, sigma_pos[2])])
			noise_radius = np.random.normal(0.0, sigma_radius)
			self.pos[i] = self.pos_init[i] + noise_pos
			self.radius[i] = self.radius_init[i] + noise_radius
			self.hit[i] = False
	def render(self):
		glColor4d(0.8, 0.1, 0.1, 0.5)
		for i in range(self.size):
			if self.hit[i]:
				continue
			p = self.pos[i]
			r = self.radius[i]
			glPushMatrix()
			glTranslated(p[0],p[1],p[2])
			glutSolidSphere(r, 20, 20)
			glPopMatrix()

