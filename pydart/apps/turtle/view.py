from OpenGL.GL import *
from OpenGL.GLU import * 
from OpenGL.GLUT import *
from OpenGL.GL.ARB.depth_texture import *
from OpenGL.GL.ARB.shadow import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
from scipy import ndimage
from PIL import Image
import math
import os
import time
import traceback
import Image

class View:
	def __init__(self, w=100, h=100, fov=45.0):
		self.w = w
		self.h = h
		self.fov = fov
		self.v_pos = np.array([0,0,0])
		self.v_look = np.array([0,0,10])
		self.v_up = np.array([0,1,0])
		self.fbo = None
		self.texture = None
		self.image = None
	def setup_texture(self):
		if not glBindFramebuffer:
			print('Missing required extensions!')
			sys.exit( testingcontext.REQUIRED_EXTENSION_MISSING )
		self.fbo = glGenFramebuffers(1)
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		self.texture = glGenTextures(1)
		glBindTexture(GL_TEXTURE_2D, self.texture)
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, self.w, self.h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, None)
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, self.texture, 0)
		glBindFramebuffer(GL_FRAMEBUFFER, 0)
	def update(self):
		glDepthFunc(GL_LEQUAL)
		glEnable(GL_DEPTH_TEST)
		glPushAttrib(GL_VIEWPORT_BIT)

		glBindTexture(GL_TEXTURE_2D, self.texture)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST) 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP) 
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
		glBindTexture(GL_TEXTURE_2D, 0) 
		glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
		glViewport(0,0,self.w,self.h)
		glDrawBuffer(GL_NONE)
		try: 
			checkFramebufferStatus( ) 
		except Exception, err: 
			traceback.print_exc() 
			os._exit(1)
		glClear(GL_DEPTH_BUFFER_BIT)
		glMatrixMode(GL_PROJECTION)
		glPushMatrix()
		glLoadIdentity()
		gluPerspective(45.0, 1.0, 1.0, 1000.0)
		glTranslatef(0,0, -10)

		glMatrixMode( GL_MODELVIEW )
		glPushMatrix()
		glLoadIdentity()
		glTranslatef(0,2.0,0)
		glutSolidSphere(1.0, 10, 10)
		glPopMatrix()

		glMatrixMode(GL_PROJECTION)
		glPopMatrix()
		glMatrixMode( GL_MODELVIEW )

		data = glReadPixels(0, 0, self.w, self.h,  GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE)
		self.image = Image.fromstring('L', (self.w, self.h), data)
		self.image = self.image.transpose(Image.FLIP_TOP_BOTTOM)
		self.image.save("test.png")
		
		glBindFramebuffer(GL_FRAMEBUFFER, 0)
		glDrawBuffer(GL_BACK)
		glPopAttrib()
		self.test()
	def test(self):
		print 'a'
	# def get_depth_image(self):
	# 	glViewport(0, 0, self.w, self.h)
	    
	#     glMatrixMode(GL_PROJECTION)
	#     glPushMatrix()
	#     glLoadIdentity()
	#     gluPerspective(self.fov, self.w / self.h, 0.1, 100.0)
	    
	#     glMatrixMode(GL_MODELVIEW)
		
	# 	glPushMatrix()

	# 	glPushAttrib(GL_ALL_ATTRIB_BITS)
	# 	glPopAttrib(GL_ALL_ATTRIB_BITS)
	def set_fov(self, fov):
		self.fov = fov
	def set_size(self, w, h):
		self.w = w
		self.h = h