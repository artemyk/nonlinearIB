import numpy as np
import tensorflow as tf


def Gaussian_entropy(d, log_var):
    # Entropy of a Gaussian distribution with 'd' dimensions and log variance 'log_var'
    h = 0.5 * d * (tf.cast(tf.log(2.0 * np.pi * np.exp(1)), tf.float32) + log_var)
    return h


def GMM_entropy(dist, log_var, d, bound='upper'):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    n = tf.cast(tf.shape(dist)[0], tf.float32)
    var = tf.exp(log_var) + 1e-6

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


def GMM_negative_LLH(dist, log_var, d):
    # computes the leave-one-out log likelihood of the log variance of a homoscedastic GMM
    # dist: a matrix of pairwise distances (tf.placeholder)
    # log_var: the log variance of a GMM (tf.variable)
    # d: number of dimensions
    # n: number of data points
    n = tf.cast(tf.shape(dist)[0], tf.float32)
    var = tf.exp(log_var) + 1e-6

    dist_norm = -(dist + 1e10 * tf.eye(tf.cast(n, tf.int32))) / (2.0 * var)  # add a large number to the diagonal elements to implement 'leave-one-out'
    const = -n * tf.log(n - 1) - 0.5 * n * d * tf.log(2.0 * np.pi * var)
    llh = const + tf.reduce_sum(tf.reduce_logsumexp(dist_norm, 1))
    return -llh


def pairwise_distance(x):
    # returns a matrix where each element is the distance between each pair of rows in x
    xx = tf.reduce_sum(tf.square(x), 1, keepdims=True)
    dist = xx - 2.0 * tf.matmul(x, tf.transpose(x)) + tf.transpose(xx)
    return dist