__author__ = 's'

import os
import sys
#sys.path.append("c:/tdm-gcc-64/bin/")

print os.environ['path']
import theano
import theano.tensor as T
print theano.config.device

import numpy
import scipy
from scipy import ndimage
import numpy as np
from scipy import misc

from PIL import Image


import numpy
import scipy
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'



I0color = misc.imread('../data/j/j_frm_00000.png')
I1color = misc.imread('../data/j/j_frm_00001.png')

def rgb2gray(rgb):
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    return np.dot(rgb[...,:3], [0.2989, 0.5870,    0.1140]).round()  #matlab's version


I0 = rgb2gray(I0color)
I1 = rgb2gray(I1color)

import scipy.misc
#I0 = scipy.misc.imresize(I0, 2.25)
#I1 = scipy.misc.imresize(I1, 2.25)
print I0.shape

I0 = I0 / 255.
I1 = I1 / 255.0

#plt.ioff()
plt.figure(figsize=(10, 10))
plt.subplot(2,1,1)
plt.imshow(I0,vmin=0, vmax=1)
plt.subplot(2,1,2)
plt.imshow(I1,vmin=0, vmax=1)
#plt.draw()
plt.show(block=False)
#plt.show()

from multiscale_of import *

pyrfactor = .7
pyrlevels = 4

of = MultiscaleOF(I0, I1, pyrlevels=pyrlevels, pyrfactor=pyrfactor, num_warps = 10,
                  train_function_args=dict(rate=0.1, num_steps = 50, alpha = 0.1,),
                  class_train_function=TrainFunctionSimple)


from time import time
t0 = time()
nu, nv = of.process()
print time() - t0


print "Finish"