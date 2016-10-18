from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from OpenGL.GL.ARB.depth_texture import *
from OpenGL.GL.ARB.shadow import *
from OpenGL.GL.framebufferobjects import *
import numpy as np
import mmMath
import myworld
from PIL import Image, ImageMath
import math
import os
import time
import traceback

EPS = 1E-6


class Eye:
    def __init__(self, w=64, h=64, fov=120.0, near=0.5, far=10, render_func=None):
        self.w = w
        self.h = h
        self.fov = fov
        self.aspect = float(w) / float(h)
        self.near = near
        self.far = far
        self.render_func = render_func
        self.frame = mmMath.I_SE3()
        self.fbo = None
        self.texture = None
        self.image = None
        self.setup = False

    def setup_texture(self):
        if self.setup:
            return
        if not glBindFramebuffer:
            print('Missing required extensions!')
            sys.exit(testingcontext.REQUIRED_EXTENSION_MISSING)
        self.fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
            self.w, self.h, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, None)
        glFramebufferTexture2D(
            GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D,
            self.texture, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self.setup = True

        print self.fbo, self.texture

    def update(self, frame=None):
        if self.render_func is None:
            return
        if self.setup is False:
            self.setup_texture()
        if frame is not None:
            self.set_frame(frame)
        # Setup gl functions for rendering depth buffer
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
        glViewport(0, 0, self.w, self.h)
        glDrawBuffer(GL_NONE)
        # try:
        #     checkFramebufferStatus()
        # except Exception, err:
        #     traceback.print_exc()
        #     os._exit(1)
        glClear(GL_DEPTH_BUFFER_BIT)
        # Projection matrix setup
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        # gluPerspective(self.fov, self.aspect, self.near, self.far)
        glOrtho(-5, 5, -5, 5, self.near, self.far)
        # Modelview matrix setup
        # Calcuate camera transformation from current frame
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        R, p = mmMath.T2Rp(self.frame)
        axis = mmMath.logSO3(R)
        angle = np.linalg.norm(axis)
        if angle > EPS:
            axis = axis / angle
        glTranslatef(p[0], p[1], p[2])
        glRotatef(180.0, 0, 1, 0)
        glRotatef(mmMath.DEG * angle, axis[0], axis[1], axis[2])
        if self.render_func is not None:
            self.render_func()
        glPopMatrix()

        glMatrixMode(GL_PROJECTION)
        glPopMatrix()

        data = glReadPixels(
            0, 0, self.w, self.h, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE)
        self.image = Image.fromstring(
            'L', (self.w, self.h), data).transpose(Image.FLIP_TOP_BOTTOM)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glDrawBuffer(GL_BACK)
        glPopAttrib()

    def test(self):
        print 'a'

    def set_frame(self, frame):
        self.frame = frame

    def save_image(self, filename):
        self.image.save(filename)

    def get_image(self, frame=None):
        if frame is not None:
            self.update(frame)
        im = ImageMath.eval("float(a)", a=self.image)
        im = ImageMath.eval("1.0-a/255.0", a=im)
        return np.reshape(im, [self.w, self.h, 1])

    def get_image_size(self):
        return (self.w, self.h)

    # def get_depth_image(self):
    #     glViewport(0, 0, self.w, self.h)
    #     glMatrixMode(GL_PROJECTION)
    #     glPushMatrix()
    #     glLoadIdentity()
    #     gluPerspective(self.fov, self.w / self.h, 0.1, 100.0)
    #     glMatrixMode(GL_MODELVIEW)
    #     glPushMatrix()
    #     glPushAttrib(GL_ALL_ATTRIB_BITS)
    #     glPopAttrib(GL_ALL_ATTRIB_BITS)

    def set_fov(self, fov):
        self.fov = fov

    def set_size(self, w, h):
        self.w = w
        self.h = h

    def render_callback(self):
        # if self.world is not None:
        #     self.world.render()
        if self.scene is not None:
            self.scene.render()
