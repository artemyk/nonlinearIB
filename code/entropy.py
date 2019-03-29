import numpy as np
import tensorflow as tf
from randomgen import RandomGenerator, MT19937
import scipy
rnd = RandomGenerator(MT19937())


def Gaussian_entropy(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (tf.cast(tf.log(2.0 * np.pi * np.exp(1)), tf.float32) + tf.log(var))
    return h

def Gaussian_entropy_np(d, var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * np.log(2.0 * np.pi * np.exp(1)) + np.log(var)
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


def get_mc_entropy(mx, var):
    n, d     = mx.shape
    mx2 = mx + rnd.standard_normal(mx.shape, dtype=np.float32)*np.sqrt(var)

    dist_norm = pairwise_distance2_np(mx, mx2) / (-2.0 * var)  
    
    const     =  - 0.5  * d * np.log(2.0 * np.pi * var) - np.log(n)
    logprobs  = const + scipy.special.logsumexp(dist_norm , axis=0) 

    return -np.mean(logprobs)
