# coding: utf-8


import numpy
import scipy
from scipy import ndimage
import numpy as np
from scipy import misc

from PIL import Image
from pyinterp2.interp2 import interp2linear


# In[36]:

import numpy
import scipy
#noinspection PyPep8Naming
import matplotlib.pyplot as plt



# In[38]:

def rgb2gray(rgb):
    #return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]).round()  #matlab's version


class MyStruct:
    def __init__(self):
        self.ololo = 1


#TODO: check for several sizes
#noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming
def extend(I, shape):
    """
    Extends image I to shape
    """
    r = [0, 0]
    Inew = I
    for i in [0, 1]:
        r = numpy.ones(Inew.shape[i], dtype='int32')
        r[-1] = shape[i] - Inew.shape[i] + 1
        Inew = np.repeat(Inew, r, axis=i)
    return Inew

#TODO: add antialiasing to work like matlab
#noinspection PyPep8Naming
def my_resize(I, factor):
    if factor != 0.5:
        raise Exception("Unsupported resize factor!")
    newshape = numpy.round((numpy.array(I.shape) ) / 2.).astype('int32')
    #print "new shape = ", newshape
    #print I[0::2,0::2].shape
    #print I[0::2,1::2].shape
    #print I[1::2,0::2].shape
    #print I[1::2,1::2].shape
    res = (
              extend(I[0::2, 0::2], newshape) +
              extend(I[0::2, 1::2], newshape) +
              extend(I[1::2, 0::2], newshape) +
              extend(I[1::2, 1::2], newshape)) / 4
    return res


import scipy.signal


def median_filter_symmetric(x, r):
    border = r
    padded = numpy.pad(x, pad_width=border, mode='symmetric')
    filt = scipy.signal.medfilt2d(padded, r)
    cropped = filt[border:-border, border:-border]
    return cropped


# In[60]:

from scipy.sparse import coo_matrix


def makeweights_helper(abs_diff, params, ):
    epsilon = 1e-5
    vd_min = numpy.min(abs_diff)
    vd_max = numpy.max(abs_diff)
    d = vd_max - vd_min
    if d == 0:
        d = 1.

    t = (abs_diff - vd_min) / d
    weights_new = numpy.exp(-(params.beta * t)) + epsilon
    weights1 = params.nu + (1 - params.nu) * weights_new

    #return numpy.ones(abs_diff.shape)
    return weights1


def makeweights(I0_, params):
    args = dict(order='F')
    M, N = I0_.shape
    MN = M * N
    X, Y = np.meshgrid(np.arange(N), np.arange(M))
    #print X, Y
    abs_diff = numpy.abs(I0_[:, :-1] - I0_[:, 1:])
    weights = makeweights_helper(abs_diff, params)
    coords1 = numpy.vstack([Y[:, :-1].ravel(**args), X[:, :-1].ravel(**args)])
    coords2 = numpy.vstack([Y[:, 1:].ravel(**args), X[:, 1:].ravel(**args)])
    #print coords1
    edges1 = np.ravel_multi_index(coords1, I0_.shape, **args)
    edges2 = np.ravel_multi_index(coords2, I0_.shape, **args)

    D1 = coo_matrix(
        (
            numpy.hstack([-weights.ravel(**args), weights.ravel(**args)]),
            (numpy.hstack([edges1, edges1]), numpy.hstack([edges1, edges2])),
        ),
        shape=(MN, MN)
    )

    abs_diff = numpy.abs(I0_[:-1, :] - I0_[1:, :])
    weights = makeweights_helper(abs_diff, params)
    coords1 = numpy.vstack([Y[:-1, :].ravel(**args), X[:-1, :].ravel(**args)])
    coords2 = numpy.vstack([Y[1:, :].ravel(**args), X[1:, :].ravel(**args)])
    edges1 = np.ravel_multi_index(coords1, I0_.shape, **args)
    edges2 = np.ravel_multi_index(coords2, I0_.shape, **args)

    D2 = coo_matrix(
        (
            numpy.hstack([-weights.ravel(**args), weights.ravel(**args)]),
            (numpy.hstack([edges1, edges1]), numpy.hstack([edges1, edges2])),
        ),
        shape=(MN, MN)
    )

    return D1, D2,

#%% Computes Lipschitz constant
def compute_lipschitz_constant(normAtA, alpha, mu_data, mu_tv):
    L = 8 * alpha / mu_tv + normAtA / mu_data
    return L


#%% Computes the derivative of the TV term
def df_tv(x, x0, mu, Dt, D1, D2):
    #%% see Eq. 17 at Ayvaci, Raptis, Soatto, NIPS'10
    x = x + x0

    D1x = D1.dot(x)
    D2x = D2.dot(x)
    tvx = (D1x * D1x + D2x * D2x) ** .5
    #print "tvx", tvx.T
    #print "mu", mu
    w = numpy.maximum(tvx, mu)
    #print "w", w.T
    u1 = D1x / w
    u2 = D2x / w

    df = Dt.dot(numpy.vstack([u1, u2]))

    f = tvx.sum()

    return df, f

#%% Computes the derivative of the Huber-L1 norm on x
def huber_l1(x, mu):
    inds = x <= mu
    #print ~inds
    fx = numpy.zeros(x.shape)
    fx[inds] = x[inds] ** 2 / (2 * mu)
    fx[~inds] = numpy.abs(x[~inds]) - (mu / 2)
    #print "fx", fx
    f = numpy.sum(fx)

    return f

# Computes the derivative of the Huber-L1 norm on the data term |Ax + b|
def df_huber_l1_Axplusb(x, A, b, mu):
    Axplusb = A.dot(x) + b
    #print "Axplusb", Axplusb
    Axbplus_abs = numpy.abs(Axplusb)
    #print "Axbplus_abs", Axbplus_abs
    #print mu
    max_ = numpy.maximum(Axbplus_abs, mu)
    #print max_
    #print "Axplusb / max_", Axplusb / max_,
    df = A.T.dot(Axplusb / max_)
    f = huber_l1(Axplusb, mu)
    return df, f

# Computes the derivative
def compute_df(x, A, b, u0, v0, alpha, mu_data, mu_tv, Dt, D1, D2):
    MN = x.shape[0] / 2
    MN2 = 2 * MN

    df1, f1 = df_huber_l1_Axplusb(x, A, b, mu_data)
    #print "df1", df1
    df2, f2 = df_tv(x[0: MN], u0, mu_tv, Dt, D1, D2)
    #print "df2", df2, df2.shape, f2.shape
    df3, f3 = df_tv(x[MN:MN2], v0, mu_tv, Dt, D1, D2)
    #print "df3", df3
    df = df1 + alpha * numpy.vstack([df2, df3])
    f = f1 + alpha * (f2 + f3)
    #print "df2", df2.T

    return df, f


def visualize(k, x, stats, A, b, u0, v0, M, N):
    #print x.shape, u0.shape, v0.shape
    MN = M * N
    u = (x[0:MN] + u0 ).reshape([N, M]).T
    v = (x[MN:] + v0 ).reshape([N, M]).T
    plt.figure(100)
    #plt.quiver( u, v, scale_units='xy', scale=0.1, angles='xy')
    plt.quiver(u, v, scale_units='xy', angles='xy')
    plt.xlim(-1, M + 1)
    plt.ylim(-1, N + 1)


# pyramid generation and flow estimation
def Huber_L1_wTV_nesterov_core(A, b, u0_, v0_, D1, D2, M, N, params):
    MN = M * N
    MN2 = MN * 2

    At = A.T
    Atb = At.dot(b)
    AtA = At.dot(A)
    AAt = A.dot(At)

    normAtA = max(AAt.diagonal())

    D1 = D1
    D2 = D2
    Dt = scipy.sparse.vstack([D1, D2]).T

    # %% Parameters
    alpha = params.alpha
    mu_data = params.mu_data
    mu_tv = params.mu_tv

    # %% Initialize
    x0 = numpy.zeros([MN2, 1])
    xk = x0
    xold = xk

    L = compute_lipschitz_constant(normAtA, alpha, mu_data, mu_tv)

    # initialize statistics storage
    stats = MyStruct()
    stats.f = numpy.zeros([1, params.maxiters])
    stats.energy = numpy.zeros([params.maxiters])
    stats.conver = numpy.zeros([params.maxiters])

    k = 0
    iteration = 1
    stop = False
    wdf = 0

    while not stop and iteration < params.maxiters:
    #if 0:
    #    print "iter", iter

        # step (1) compute the derivative
        df, f = compute_df(xk, A, b, u0_, v0_, alpha, mu_data, mu_tv, Dt, D1, D2)
        # step (2) update yk
        yk = xk - (1 / L) * df

        # step (3) update zk
        alphak = (k + 1) / 2.
        wdf += alphak * df
        zk = x0 - (1 / L) * wdf

        # step (4) blend yk and zk
        tauk = 2. / (k + 3)
        xkp = tauk * zk + (1 - tauk) * yk
        xk = xkp

        # save statistics
        stats.energy[iteration] = f

        if iteration > 10:
            iterm10 = iteration - 10
            fbar = numpy.mean(stats.energy[iterm10:iteration])
            convergence = abs(f - fbar) / fbar
            #print "convergence", convergence
            if convergence < 1e-4:
                stop = True



        # visualize
        if params.display and (((k + 1) % 100) == 1 or stop or ((iteration + 1) == params.maxiters)):
        #if 1:
            visualize(k, xk, stats, A, b, u0_, v0_, M, N)

        xold = xk
        k += 1
        iteration += 1

    xk = xk + numpy.vstack([u0_, v0_])
    return xk


# In[116]:

#np.set_printoptions(precision=5, linewidth = 250)
##At.todense()
##print L, normAtA, alpha, params.alpha, df
#df, f = compute_df(xk, A, b, u0_, v0_, alpha, mu_data, mu_tv, Dt, D1, D2);

def Huber_L1_wTV_nesterov(I0, I1warped, u0, v0, Ix, Iy, It, params):
    M, N = I0.shape
    b = It.T.reshape([-1, 1])
    A = scipy.sparse.hstack([
        scipy.sparse.diags([Ix.T.ravel()], [0]),
        scipy.sparse.diags([Iy.T.ravel()], [0]),
        ])
    D1, D2 = makeweights(I0, params)
    x = Huber_L1_wTV_nesterov_core(A, b, u0.reshape(-1, 1, order='F'), v0.reshape(-1, 1, order='F'), D1, D2, M, N,
                                   params)
    x = x.reshape([2, M * N]).T
    u = x[:, 0].reshape([N, M]).T
    v = x[:, 1].reshape([N, M]).T
    return u, v


import sys


def construct_image_pyramid(I, pyrlevels, pyrfactor):
    #print >> sys.stderr, "Pyramid is constructed to [0,pyrlevels-1] range!!"
    factor = 2. ** .5
    smooth_sigma = (1. / pyrfactor) ** .5 / factor

    pyr = []
    tmp = I
    pyr.append(I)
    for m in range(pyrlevels - 1):
        filt1 = scipy.ndimage.filters.gaussian_filter(tmp, smooth_sigma, order=0, output=None, mode='reflect', cval=0.0,
                                                      truncate=2.0)
        #filt1r  = scipy.ndimage.zoom(filt1, pyrfactor, order=1)
        filt1r = my_resize(filt1, pyrfactor)
        tmp = filt1r
        pyr.append(filt1r)
    return pyr


def Huber_L1_wTV_nesterov_pyramid(I0, I1):
    params = MyStruct()
    pyrfactor = .5
    warps = 5
    pyrlevels = 2
    # ALPHA is the coefficient of the regularizer. When the option do_varying_alpha is selected,
    # for each pyramid level, its value varies between the values alpha0 and alphamax with the
    # multiplier alphamult at each warping step.
    params.do_varying_alpha = True
    if params.do_varying_alpha:
        params.alpha0 = 0.006
        params.alphamult = 5
        params.alphamax = 0.8
    else:
        params.alpha = 0.2


    # Thresholds for Huber-L1 norm for data term and regularizer
    params.mu_tv = 0.01
    params.mu_data = 0.01

    # Parameters of weights for gradients: w(x) = NU - (1-NU) exp(-BETA |\nabla I(x)|^2_2)
    params.beta = 30
    params.nu = 0.01

    params.maxiters = 500  # max number of iterations for each optimization loop
    params.display = True     # display results

    iscolor = False

    I0pyr = construct_image_pyramid(I0, pyrlevels, pyrfactor)
    I1pyr = construct_image_pyramid(I1, pyrlevels, pyrfactor)

    import scipy.ndimage

    #level = pyrlevels - 1
    for level in range(pyrlevels - 1, -1, -1):
        M, N = I0pyr[level].shape
        if level == pyrlevels - 1:
            u = numpy.zeros_like(I0pyr[level])
            v = numpy.zeros_like(I0pyr[level])
        else:
            # rescale motion vector field and length of the vectors
            rescale_v, rescale_u = numpy.array([M, N], dtype='float32') / I0pyr[level + 1].shape
            #rescale_u = float(N) / I0pyr[level+1].shape[1]
            #rescale_v = float(M) / I0pyr[level+1].shape[0]

            u = scipy.ndimage.zoom(u, [rescale_v, rescale_u], order=3) * rescale_u
            v = scipy.ndimage.zoom(v, [rescale_v, rescale_u], order=3) * rescale_v
            if u.shape != I0pyr[level].shape or v.shape != I0pyr[level].shape:
                raise Exception("Resize failed during transition to higher levels. Need better resize implementation.")

        u0 = u
        v0 = v
        I0_ = I0pyr[level]
        I1_ = I1pyr[level]

        idx, idy = np.meshgrid(np.arange(N), np.arange(M))   # indexing from 0 !!

        # Compute the spatial derivatives
        mask = numpy.array([1, -8, 0, 8, -1], ndmin=2) / 12.0
        Ix = ndimage.correlate(I0_, mask, mode='nearest') #Ix = imfilter(I0_, mask, 'replicate');
        Iy = ndimage.correlate(I0_, mask.T, mode='nearest') #Iy = imfilter(I0_, mask','replicate');

        if params.do_varying_alpha:
            params.alpha = params.alpha0

        for i in range(warps):
            if params.display:
                print 'Pyramid level %d, Warp %d' % (level, i)

            # Median filtering of the motion field
            u0 = median_filter_symmetric(u, 5)  #medfilt2(u, [5 5], 'symmetric');
            v0 = median_filter_symmetric(v, 5)  # medfilt2(v, [5 5], 'symmetric');
            #e0 = numpy.zeros([M,N]);

            idxx = idx + u0
            idyy = idy + v0

            I1warped = interp2linear(I1_, idxx, idyy)
            It = I1warped - I0_

            # boundary handling
            m = (idxx > N - 1) | (idxx < 0) | (idyy > M - 1) | (idyy < 0)
            Ix[m] = 0.0
            Iy[m] = 0.0
            It[m] = 0.0

            # Estimate the motion from I0 to warped I
            u, v = Huber_L1_wTV_nesterov(I0_, I1warped, u0, v0, Ix, Iy, It, params)

            if params.do_varying_alpha:
                params.alpha = min(params.alphamult * params.alpha, params.alphamax)

    return u, v


def main(argv):
    I0color = misc.imread('../data/j/j_frm_00000.png')
    I1color = misc.imread('../data/j/j_frm_00001.png')

    I0 = rgb2gray(I0color)
    I1 = rgb2gray(I1color)

    #I0 = numpy.array([ 0, 1, 0, 0, 2, 0, 0, 1, 0]).reshape(3,3) * 255 / 2.0;
    #I1 = numpy.array([ 1, 0, 0, 2, 0, 0, 1, 0, 0]).reshape(3,3) * 255 / 2.0;

    #I0 = numpy.array([ 0, 1, 0, 0, 2, 0, 0, 1, 0, 2, 0, 0]).reshape(4,3) * 255 / 2.0;
    #I1 = numpy.array([ 1, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 0]).reshape(4,3) * 255 / 2.0;



    I0 /= 255.
    I1 /= 255.0

    u, v = Huber_L1_wTV_nesterov_pyramid(I0, I1)

    import scipy.io

    scipy.io.savemat('python_weighted__.mat', {'u': u, 'v': v})
    print "done"


if __name__ == "__main__":
    main(sys.argv)
