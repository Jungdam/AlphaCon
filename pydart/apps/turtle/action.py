from abc import ABCMeta, abstractmethod
import numpy as np
import basics
import time


class ActionBase:
    def __init__(self, dim, randseed=None):
        self.dim = dim
        self.val_def = np.zeros(self.dim)
        self.val_min = None
        self.val_max = None
        if randseed is None:
            randseed = int(time.time() * 10000) % 10000
        self.rand = np.random.RandomState(randseed)
        self.initialize()

    @abstractmethod
    def initialize(self):
        raise NotImplementedError("Must override")

    def clamp(self, a):
        act = np.array(a)
        if self.val_min is None or self.val_max is None:
            return act
        for i in range(self.dim):
            if act[i] < self.val_min[i]:
                act[i] = self.val_min[i]
            if act[i] > self.val_max[i]:
                act[i] = self.val_max[i]
        return act

    def check_range(self, a):
        if self.val_min is None or self.val_max is None:
            return True
        for i in range(self.dim):
            if not self.val_min[i] < a[i] < self.val_max[i]:
                return False
        return True

    def random(self, sigma, apply_clamp=True):
        act = self.rand.normal(self.val_def, sigma)
        if apply_clamp:
            return self.clamp(act)
        else:
            return act

    def delta(self, a):
        return a - self.val_def

    def add(self, a):
        return a + self.val_def


default = np.array([
    -1.25567921, 0.6118376, 0.53513041,
    0.28105493, 0.78491477, -0.65140349,
    -1.25567921, 0.6118376, -0.53513041,
    -0.28105493, -0.78491477, 0.65140349,
    1.5])


val_min = np.array([
    -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
    -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
    1.0])


val_max = np.array([
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
    2.0])

# default = np.array([\
#   0, 0, 0, 0, 0, 0,
#   0, 0, 0, 0, 0, 0,
#   1.5])

# val_min = np.array([\
#   -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
#   -2.0, -2.0, -2.0, -2.0, -2.0, -2.0,
#   0.5])

# val_max = np.array([\
#   2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
#   2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
#   2.5])

dim_left = 6
dim_right = 6
dim_others = 1
dim = dim_left + dim_right + dim_others


def get_left(a):
    return a[0:dim_left]


def get_right(a):
    return a[dim_left:dim_left + dim_right]


def get_time(a):
    return a[-1]


def mirror(a):
    return np.array([a[0], a[1], -a[2], -a[3], -a[4], -a[5]])


def zero():
    np.zeros(dim())


def pprint(a):
    print np.array(a[0]), np.array(a[1]), np.array([a[2]])


def clamp(a):
    act = np.array(a)
    for i in range(dim):
        if act[i] < val_min[i]:
            act[i] = val_min[i]
        if act[i] > val_max[i]:
            act[i] = val_max[i]
    return act


def check_range(a):
    for i in range(dim):
        if not val_min[i] < a[i] < val_max[i]:
            # print 'min', a[i], val_min[i], a[i]<val_min[i]
            # print 'max', a[i], val_max[i], a[i]>val_max[i]
            return False
    return True


def random(mu, sigma, apply_clamp=True):
    act = np.random.normal(mu, sigma)
    if apply_clamp:
        return clamp(act)
    else:
        return act
