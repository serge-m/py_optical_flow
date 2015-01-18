__author__ = 'Sergey Matyunin'


import numpy as np
import numpy
from pyinterp2.interp2 import interp2linear
from scipy import ndimage
from train_function import TrainFunctionSimple
import theano 

class Warper:
    def __init__(self, shape, u0, v0, I0, I1, train_function = None, display = False, intermediate_saver = None):
        """
            shape - shape of input function,
            u0, v0 - starting values of flow
            I0, I1 - images to compute flow between

        """
        self.M, self.N = shape[0], shape[1]
        self.u, self.v = u0.copy().astype(theano.config.floatX), v0.copy().astype(theano.config.floatX)
        self.idx, self.idy = np.meshgrid(np.arange(self.N), np.arange(self.M))
        self.mask = np.array([1, -8, 0, 8, -1], ndmin=2, dtype=theano.config.floatX)/12.0
        self.display = display
        self.counter = 0
        self.I0, self.I1 = I0.copy().astype(theano.config.floatX), I1.copy().astype(theano.config.floatX)
        self._intermediate_saver = intermediate_saver

        if train_function is None:
            train_function = TrainFunctionSimple(u0, v0, rate=0.1, num_steps = 120)
            
        self.train = train_function

    # I0 - compensate to it
    def warp(self):
        if self.display:
            print 'Warp %d' % (self.counter,)

        u0, v0 = self.u.copy(), self.v.copy()

        idxx = self.idx + u0
        idyy = self.idy + v0

        idxx = np.clip(idxx, 0, self.N-1)
        idyy = np.clip(idyy, 0, self.M-1)

        I1warped = interp2linear(self.I1, idxx, idyy).astype(theano.config.floatX)
        if self.display:
            #plt.figure()
            #plt.imshow(I1warped)
            print "I1warped", I1warped
            print "I0", self.I0
            print "u0", u0
            print "v0", v0
            pass

        It = I1warped - self.I0
        if self.display:
            print "It", It
            
        Ix = ndimage.correlate(I1warped, self.mask, mode='nearest')
        Iy = ndimage.correlate(I1warped, self.mask.T, mode='nearest')

        # boundary handling
        m = (idxx > self.N - 1) | (idxx < 0) | (idyy > self.M - 1) | (idyy < 0)
        Ix[m] = 0.0
        Iy[m] = 0.0
        It[m] = 0.0

        self.Ix = Ix
        self.Iy = Iy
        self.It = It

        self.train.init(np.zeros_like(self.I0), np.zeros_like(self.I0))
        cnt_step = 0
        while not self.train.done():
            e = self.train.step(Ix, Iy, It)[0]
            if self.display and (cnt_step % 10 == 0 or self.train.done()):
                print e,
            cnt_step += 1



        self.u += self.train.tu.get_value()
        self.v += self.train.tv.get_value()
        self.counter += 1

        if self._intermediate_saver:
            self._intermediate_saver.save_members(self)
            self._intermediate_saver.save_locals(locals())
