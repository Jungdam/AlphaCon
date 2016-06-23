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

class Scene1(SceneBase):
	def __init__(self, skel, pos_radius_init, pos_radius_sigma=[[3.0, 3.0, 3.0],0.01]):
		self.skel = skel
		self.pos_init = pos_radius_init[0]
		self.radius_init = pos_radius_init[1]
		self.pos = self.pos_init
		self.radius = self.radius_init
		self.sigma_pos = pos_radius_sigma[0]
		self.sigma_radius = pos_radius_sigma[1]
	def get_pos(self):
		return self.pos
	def score(self):
		R,p = mmMath.T2Rp(self.skel.body('trunk').T)
		axis = R[:,2]
		diff = self.pos-p
		# lengh difference
		l = np.linalg.norm(diff)
		# angle difference
		a = np.dot(axis,diff/l)
		return -l,a
	def update(self):
		return None
	def perturbate(self):
		noise_pos = np.array([
			np.random.uniform(-self.sigma_pos[0], self.sigma_pos[0]),
			np.random.uniform(-self.sigma_pos[1], self.sigma_pos[1]),
			np.random.uniform(-self.sigma_pos[2], self.sigma_pos[2])])
		noise_radius = np.random.normal(0.0, self.sigma_radius)
		self.pos = self.pos_init + noise_pos
		self.radius = self.radius_init + noise_radius
	def render(self):
		color = [1.0, 0.1, 0.1, 0.7]
		glColor4d(color[0],color[1],color[2],color[3])
		p = self.pos
		r = self.radius
		glPushMatrix()
		glTranslated(p[0],p[1],p[2])
		glutSolidSphere(r, 20, 20)
		glPopMatrix()

class Scene(SceneBase):
	def __init__(self, skel, pos=None, radius=None):
		self.skel = skel
		self.pos_init = []
		self.radius_init = []
		self.pos = []
		self.radius = []
		self.hit = []
		self.size = 0
		self.hit_radius = 0.3
		if pos != None and radius != None:
			self.generate([pos, radius])
	def get_pos(self):
		return self.pos[0]
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
	def perturbate(self, data=[[3.0, 3.0, 8.0],0.01]):
		sigma_pos=data[0]
		sigma_radius=data[1]
		for i in range(self.size):
			z = np.random.uniform(2.0, sigma_pos[2])
			frac = z/sigma_pos[2]
			x = frac*np.random.uniform(-sigma_pos[0], sigma_pos[0])
			y = frac*np.random.uniform(-sigma_pos[0], sigma_pos[0])
			noise_pos = np.array([x,y,z])
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

