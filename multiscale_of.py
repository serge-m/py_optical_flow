__author__ = 'Sergey Matyunin'

import numpy
import numpy as np
from pyinterp2.interp2 import interp2linear
import scipy.ndimage
from train_function import TrainFunctionSimple, TrainFunctionTV
from intermediate_saver import IntermediateSaver
from warper import Warper
import theano 

def construct_image_pyramid(I, pyrlevels, pyrfactor):
    factor = 2. ** .5
    smooth_sigma = (1. / pyrfactor) ** .5 / factor

    pyr = []
    tmp = I
    pyr.append(I.copy())
    for m in range(pyrlevels - 1):
        shape = (numpy.array(tmp.shape) * pyrfactor).astype('int32')
        idx, idy = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        idx = idx *tmp.shape[1] / shape[1].astype(theano.config.floatX)
        idy = idy *tmp.shape[0] / shape[0].astype(theano.config.floatX)
        filt1 = scipy.ndimage.filters.gaussian_filter(tmp, smooth_sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=2.0)
        filt1r = interp2linear(filt1, idx, idy,).astype(theano.config.floatX)
        tmp = filt1r
        pyr.append(filt1r)
    return pyr

class MultiscaleOF:
    def __init__(self, I0, I1, pyrlevels, pyrfactor, num_warps, train_function_args=dict(rate=0.1, num_steps = 1000, alpha = 0.1), class_train_function = TrainFunctionTV):
        self.pyrlevels = pyrlevels
        self.pyrfactor = pyrfactor
        self.I0pyr = construct_image_pyramid(I0.astype(theano.config.floatX), self.pyrlevels, self.pyrfactor)
        self.I1pyr = construct_image_pyramid(I1.astype(theano.config.floatX), self.pyrlevels, self.pyrfactor)
        self.warps = num_warps
        self.u = None
        self.v = None
        self.train_function_args = train_function_args
        self.class_train_function = class_train_function

    def process(self):
        for level in range(self.pyrlevels - 1, -1, -1):
            if level == self.pyrlevels - 1:
                u0 = numpy.zeros_like(self.I0pyr[level])
                v0 = numpy.zeros_like(self.I0pyr[level])
            else:
                M, N = self.I0pyr[level].shape
                rescale_v, rescale_u = numpy.array([M, N], dtype=theano.config.floatX) / self.I0pyr[level + 1].shape
                u0 = scipy.ndimage.zoom(self.u, [rescale_v, rescale_u], order=3) * rescale_u
                v0 = scipy.ndimage.zoom(self.v, [rescale_v, rescale_u], order=3) * rescale_v
                if u0.shape != self.I0pyr[level].shape or v0.shape != self.I0pyr[level].shape:
                    raise Exception("Resize failed during transition to higher levels. Need better resize implementation.")

            self.process_level(level, u0, v0)
        return self.u, self.v

    def process_level(self, level, u0, v0):

        I0_ = self.I0pyr[level]
        I1_ = self.I1pyr[level]

        #print I1_.shape, u0.shape, self.I0pyr[level].shape
        wrpr = Warper(I0_.shape, u0, v0, I0_, I1_,
                      train_function=self.class_train_function(u0, v0, **self.train_function_args), display=0)

        for i in range(self.warps):
            wrpr.warp()
        self.u, self.v = wrpr.u, wrpr.v
