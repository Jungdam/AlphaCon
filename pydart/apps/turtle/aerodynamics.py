import math
import numpy as np

rho = 1000.0
eps = 1.0e-5

def C_D(theta_rad):
    return 2.0 * (1.0-math.fabs(math.cos(theta_rad))) + 0.005

def C_L(theta_rad):
    cutoff = 85.0

    xp = [-90,-10,30,90]
    fp = [-0.2,-0.4,2.0,0.8]

    theta = theta_rad * 180 / math.pi

    if math.fabs(theta) > cutoff:
        return 0.0
    else:
        return np.interp(theta, xp, fp)

def compute(v, n, a):

    v_norm = np.linalg.norm(v)

    # zero velocities
    if v_norm < eps:
        return np.zeros(3)
    v = -v
    normalDir = -n/np.linalg.norm(n)
    dragDir = v/v_norm

    # reverse direction
    if np.dot(normalDir, dragDir) < eps:
        return np.zeros(3)

    l = np.zeros(3)

    # when normal and drag directions coincide
    if np.linalg.norm(normalDir-dragDir) < eps:
        if normalDir[0] > 0.5 or normalDir[1] > 0.5:
            l = np.array([normalDir[1],-normalDir[0],0.0])
        else:
            l = np.array([0.0,-normalDir[2],normalDir[1]])
    else:
        l = np.cross(dragDir,normalDir)
    
    l = l/np.linalg.norm(l)
    liftDir = np.cross(l,dragDir)

    v_n = np.dot(dragDir,normalDir) * v
    v_t = v - v_n

    theta = math.atan2(np.dot(v_n,normalDir),np.linalg.norm(v_t))

    dragForce = 0.5 * rho * a * C_D(theta) * v_norm * v_norm * dragDir;
    liftForce = 0.5 * rho * a * C_L(theta) * v_norm * v_norm * liftDir;
    totalForce = dragForce + liftForce

    return totalForce

