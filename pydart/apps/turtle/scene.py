from abc import ABCMeta, abstractmethod
import numpy as np
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import mmMath

# class Object:
# 	def __init__(self):
# 		self.T = mmMath.I_SE3()
# 	def collide(self, other)

class SceneBase:
	__metaclass__ = ABCMeta
	def __init__(self):
		return
	@abstractmethod
	def score(self):
		raise NotImplementedError("Must override")
	@abstractmethod
	def update(self):
		raise NotImplementedError("Must override")
	@abstractmethod
	def perturbate(self):
		raise NotImplementedError("Must override")
	@abstractmethod
	def render(self):
		raise NotImplementedError("Must override")

class Scene(SceneBase):
	def __init__(self, skel, pos=None, radius=None):
		self.skel = skel
		self.pos_init = []
		self.radius_init = []
		self.pos = []
		self.radius = []
		self.hit = []
		self.size = 0
		self.hit_radius = 0.25
		if pos != None and radius != None:
			self.generate([pos, radius])
	def score(self):
		cnt = 0
		for i in range(self.size):
			if self.hit[i]:
				cnt += 1
		return float(cnt)
	def update(self):
		transform = self.skel.body('trunk').T
		radius = self.hit_radius
		for i in range(self.size):
			p = self.pos[i]
			r = self.radius[i]+radius
			l = np.linalg.norm(p-transform[0:3,3])
			if l < r:
				self.hit[i] = True
	def perturbate(self, data=[[2.0, 2.0, 4.0],0.01]):
		sigma_pos=data[0]
		sigma_radius=data[1]
		for i in range(self.size):
			noise_pos = np.array([
				np.random.uniform(-sigma_pos[0], sigma_pos[0]),
				np.random.uniform(-sigma_pos[1], sigma_pos[1]),
				np.random.uniform(-sigma_pos[2], sigma_pos[2])])
			noise_radius = np.random.normal(0.0, sigma_radius)
			self.pos[i] = self.pos_init[i] + noise_pos
			self.radius[i] = self.radius_init[i] + noise_radius
			self.hit[i] = False
	def render(self):
		color = [0.8, 0.1, 0.1, 0.5]
		glColor4d(color[0],color[1],color[2],color[3])
		for i in range(self.size):
			if self.hit[i]:
				continue
			p = self.pos[i]
			r = self.radius[i]
			glPushMatrix()
			glTranslated(p[0],p[1],p[2])
			glutSolidSphere(r, 20, 20)
			glPopMatrix()
	def generate(self, data=None):
		pos = data[0]
		radius = data[1]
		self.pos_init = pos[:]
		self.radius_init = radius[:]
		self.pos = pos[:]
		self.radius = radius[:]
		self.size = len(pos)
		self.hit = [False] * self.size

