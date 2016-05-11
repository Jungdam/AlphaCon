import pydart
import math
import numpy as np
import controller

class Myworld:
	def __init__(self, h, skel_file, step_callback=None):
		self.world = pydart.create_world(h, skel_file)
		self.skel = self.world.skels[0]
		self.controller = controller.Controller(self.skel, h)
		self.skel.controller = self.controller
		self.step_callback = step_callback
	def reset(self):
		self.world.reset()
		self.controller.reset()
	def step(self):
		if self.step_callback is not None:
			self.step_callback(self.world)
		self.world.step()
	def get_world(self):
		return self.world
	def get_controller(self):
		return self.controller
	def get_skeleton(self):
		return self.skel
	def get_time_step(self):
		return self.world.dt