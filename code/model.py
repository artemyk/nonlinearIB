import tensorflow as tf
import numpy as np
import entropy

class Net(object):
    def __init__(self, hiddenD, init_beta = 0.0, trainable_sigma = True, log_sigma2 = 0., log_eta2 = 0.):
        self.hiddenD = hiddenD
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784]) # digit images
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # one-hot labels
        self.trainable_sigma = trainable_sigma
        
        self.beta = tf.get_variable('beta', dtype=tf.float32, trainable=False, initializer=float(init_beta))
        self.log_eta2   = tf.get_variable('log_eta2', dtype=tf.float32, trainable=True, initializer=float(log_eta2))
        if trainable_sigma:
            self.log_sigma2 = tf.get_variable('log_sigma2', dtype=tf.float32, trainable=False, initializer=float(log_sigma2))
        else:
            self.log_sigma2 = tf.constant(float(log_sigma2), dtype=tf.float32)
            
        self.T1       = tf.layers.dense(self.x, 512, activation=tf.nn.relu)
        self.Tnonoise = tf.layers.dense(self.T1, self.hiddenD, activation=tf.nn.relu)
        sigma         = tf.exp(0.5 * self.log_sigma2)
        self.T        = self.Tnonoise + tf.random_normal(shape=tf.shape(self.Tnonoise), mean=0.0, stddev=sigma, dtype=tf.float32)

        self.D        = tf.layers.dense(self.T, 512, activation=tf.nn.relu)
        self.predY    = tf.layers.dense(self.D, 10, activation=None) # , kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.distance_matrix = entropy.pairwise_distance(self.Tnonoise)
        self.neg_llh_eta = entropy.GMM_negative_LLH(self.distance_matrix, self.log_eta2, self.hiddenD)   # negative log-likelihood for the 'width' of the GMM
        self.eta_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_llh_eta, var_list=[self.log_eta2])
        
        
        
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.predY))
        self.accuracy       = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predY, 1), tf.argmax(self.y, 1)), tf.float32))

        self.H_T = entropy.GMM_entropy(self.distance_matrix, tf.log(tf.exp(self.log_sigma2) + tf.exp(self.log_eta2)), self.hiddenD, 'upper')
        self.H_T_given_X = entropy.Gaussian_entropy(hiddenD, self.log_sigma2)
        self.Ixt = self.H_T - self.H_T_given_X

        self.Iyt = tf.log(10.0) - self.cross_entropy

        self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) # learning_rate=self.learning_rate_ph, epsilon=0.0001)
        self.loss           = self.beta*(self.Ixt**2) - self.Iyt
        self.trainstep      = self.adam_optimizer.minimize(self.loss)

        if trainable_sigma:
            self.sigma_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list=[self.log_sigma2])

