from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import math
import numpy as np
import mmMath

def render_transform(T,scale=1.0,line_width=1.0,point_size=1.0):
    glPointSize(point_size)
    glLineWidth(line_width)
    
    R,p = mmMath.T2Rp(T)
    glPushMatrix()
    glTranslated(p[0],p[1],p[2])
    glScalef(scale,scale,scale)

    glColor3d(1.0, 1.0, 0.0)
    glutSolidSphere(0.05, 10, 10)

    glBegin(GL_LINES)
    glColor3d(1,0,0)
    glVertex3d(0,0,0)
    glVertex3d(R[0,0],R[1,0],R[2,0])
    glColor3d(0,1,0)
    glVertex3d(0,0,0)
    glVertex3d(R[0,1],R[1,1],R[2,1])
    glColor3d(0,0,1)
    glVertex3d(0,0,0)
    glVertex3d(R[0,2],R[1,2],R[2,2])
    glEnd()

    glPopMatrix()
def render_ground(size=[20.0,20.0],dsize=[1.0,1.0],color=[0.0,0.0,0.0],line_width=1.0):
    lx = size[0]
    lz = size[1]
    dx = dsize[0]
    dz = dsize[1]
    nx = int(lx/dx)+1
    nz = int(lz/dz)+1

    glColor3d(color[0],color[1],color[2])
    glLineWidth(line_width)
    for i in np.linspace(-0.5*lx,0.5*lx,nx):
        glBegin(GL_LINES)
        glVertex3d(i,0,-0.5*lz)
        glVertex3d(i,0,0.5*lz)
        glEnd()
    for i in np.linspace(-0.5*lz,0.5*lz,nz):
        glBegin(GL_LINES)
        glVertex3d(-0.5*lx,0,i)
        glVertex3d(0.5*lx,0,i)
        glEnd()

    render_transform(mmMath.I_SE3())