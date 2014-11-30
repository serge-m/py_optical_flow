__author__ = 'Sergey Matyunin'


import os
import numpy

_base_path = os.path.split(os.path.realpath(__file__))[0]


def get_image(name, idx):
    if not idx in [0, 1]:
        raise Exception("Only 0 and 1 frames are supported")
    path = os.path.join(_base_path, name, "I{}.txt".format(idx))
    with open(path, 'r') as f:
        return numpy.loadtxt(f)