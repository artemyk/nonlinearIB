# Utility functions for computing entropy values

import numpy as np
import tensorflow as tf
import scipy


def gaussian_entropy(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (tf.cast(tf.log(2.0 * np.pi * np.exp(1)), tf.float32) + tf.log(var))
    return h

def gaussian_entropy_np(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (np.log(2.0 * np.pi * np.exp(1)) + np.log(var))
    return h


def GMM_entropy(dist, var, d, bound='upper'):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    n = tf.cast(tf.shape(dist)[0], tf.float32)
    # var = tf.exp(log_var) + 1e-10

    if bound is 'upper':
        dist_norm = - dist / (2.0 * var)  # uses the KL distance
    elif bound is 'lower':
        dist_norm = - dist / (8.0 * var)  # uses the Bhattacharyya distance
    else:
        print('Error: invalid bound argument')
        return 0

    const = 0.5 * d * tf.log(2.0 * np.pi * np.exp(1.0) * var) + tf.log(n)
    h = const - tf.reduce_mean(tf.reduce_logsumexp(dist_norm, 1))
    return h


def GMM_negative_LLH(dist, var, d):
    # computes the leave-one-out log likelihood of the log variance of a homoscedastic GMM
    # dist: a matrix of pairwise distances (tf.placeholder)
    # log_var: the log variance of a GMM (tf.variable)
    # d: number of dimensions
    # n: number of data points
    n = tf.cast(tf.shape(dist)[0], tf.float32)
    # var = tf.exp(log_var) + 1e-10

    dist_norm = -(dist + 1e10 * tf.eye(tf.cast(n, tf.int32))) / (2.0 * var)  # add a large number to the diagonal elements to implement 'leave-one-out'
    const = -n * tf.log(n - 1) - 0.5 * n * d * tf.log(2.0 * np.pi * var)
    llh = const + tf.reduce_sum(tf.reduce_logsumexp(dist_norm, 1))
    return -llh


def pairwise_distance(x):
    # returns a matrix where each element is the distance between each pair of rows in x
    xx = tf.reduce_sum(tf.square(x), 1, keepdims=True)
    dist = xx - 2.0 * tf.matmul(x, tf.transpose(x)) + tf.transpose(xx)
    return dist

def pairwise_distance2_np(x, x2):
    # returns a matrix where each element is the distance between each pair of rows in x and x2
    xx = np.sum(x**2, axis=1)[:,None]
    x2x = np.sum(x2**2, axis=1)[:,None]
    dist = xx + x2x.T - 2.0 * x.dot(x2.T)
    return dist


def get_gib_curve(covXY, xdims):
    # get optimal IB curve for gaussian variables
    #http://www.jmlr.org/papers/volume6/chechik05a/chechik05a.pdf

    covX = covXY[:xdims,:xdims]
    covY = covXY[xdims:,xdims:]
    covXgY = covX - covXY[:xdims,xdims:].dot(np.linalg.inv(covY)).dot(covXY[xdims:,:xdims])
    mainMx = covXgY.dot(np.linalg.inv(covX))

    evecs, evals = np.linalg.eig(mainMx)
    #print(evecs.min(), evecs.max())
    assert(np.all(evecs >= -1e-5) and np.all(evecs <= 1+1e-5))
    ix = np.argsort(evecs)
    sorted_evecs = np.real(evecs[ix])
    sorted_evecs = sorted_evecs[np.logical_not(np.isclose(sorted_evecs, 1))]
    

    v1, v2 = [], []
    for alpha in np.linspace(0., 1000, 1000):
        #last_alpha = np.flatnonzero(alpha >= 1./(1.-sorted_evecs))[-1]
        #print(last_alpha)

        Itx = 0.5*np.sum([np.log(alpha*(1-l)/l) for l in sorted_evecs if alpha >= 1./(1.-l)])
        Ity = Itx - 0.5*np.sum([np.log(alpha*(1-l)) for l in sorted_evecs if alpha >= 1./(1.-l)])

        v1.append(Itx)
        v2.append(Ity)
        
    return np.array(v1), np.array(v2)
