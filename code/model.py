import tensorflow as tf
import numpy as np
import entropy
ds = tf.contrib.distributions


class Net(object):
    def __init__(self, encoder_arch, decoder_arch, init_beta = 0.0, trainable_sigma = True, log_sigma2 = 0., log_eta2 = 0.):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 784]) # digit images
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])  # one-hot labels
        self.trainable_sigma = trainable_sigma
        
        self.beta = tf.get_variable('beta', dtype=tf.float32, trainable=False, initializer=float(init_beta))
        self.log_eta2   = tf.get_variable('log_eta2', dtype=tf.float32, trainable=False, initializer=float(log_eta2))
        if trainable_sigma:
            self.log_sigma2 = tf.get_variable('log_sigma2', dtype=tf.float32, trainable=False, initializer=float(log_sigma2))
        else:
            self.log_sigma2 = tf.constant(float(log_sigma2), dtype=tf.float32)
            
        self.encoder = [self.x,]
        for neurons, activation in encoder_arch:
            self.encoder.append(tf.layers.dense(self.encoder[-1], neurons, activation=getattr(tf.nn, activation)))
        #self.Tnonoise = tf.layers.dense(self.T1, self.hiddenD, activation=tf.nn.relu)
        sigma         = tf.exp(0.5 * self.log_sigma2)
        self.T        = self.encoder[-1] + tf.random_normal(shape=tf.shape(self.encoder[-1]), 
                                                            mean=0.0, stddev=sigma, dtype=tf.float32)
        self.decoder = [self.T,]
        for neurons, activation in decoder_arch:
            self.decoder.append(tf.layers.dense(self.decoder[-1], neurons, activation=getattr(tf.nn, activation)))
            
        self.predY    = tf.layers.dense(self.decoder[-1], 10, activation=None)
        
        hiddenD = encoder_arch[-1][0]
        
        self.distance_matrix = entropy.pairwise_distance(self.encoder[-1])
        
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.predY))
        self.accuracy       = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predY, 1), tf.argmax(self.y, 1)), tf.float32))

        self.H_T = entropy.GMM_entropy(self.distance_matrix, 
                                       tf.log(tf.exp(self.log_sigma2) + tf.exp(self.log_eta2)), hiddenD, 'upper')
        self.H_T_given_X = entropy.Gaussian_entropy(hiddenD, self.log_sigma2)
        self.Ixt = self.H_T - self.H_T_given_X

        self.Iyt = tf.log(10.0) - self.cross_entropy

        self.adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) # learning_rate=self.learning_rate_ph, epsilon=0.0001)
        self.nlIB_loss           = self.beta*(self.Ixt**2) - self.Iyt
        self.nlIB_trainstep      = self.adam_optimizer.minimize(self.nlIB_loss)

        prior = ds.Normal(0.0, 1.0)
        encoding = ds.Normal(self.encoder[-1], tf.exp(self.log_sigma2))
        self.vIxt = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(encoding, prior), 0))
        self.vIB_loss           = self.beta*(self.vIxt**2) - self.Iyt
        self.vIB_trainstep      = self.adam_optimizer.minimize(self.vIB_loss)
        
        
        if trainable_sigma:
            raise Exception('Not supported (must support for VIB also)')
            self.sigma_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, var_list=[self.log_sigma2])

        # negative log-likelihood for the 'width' of the GMM
        self.neg_llh_eta = entropy.GMM_negative_LLH(self.distance_matrix, self.log_eta2, hiddenD)   
        self.eta_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_llh_eta, var_list=[self.log_eta2])
        
