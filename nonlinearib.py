# Implementation of 
# A Kolchinsky, BD Tracey, DH Wolpert, "Nonlinear Information Bottleneck", https://arxiv.org/abs/1705.02436
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer
from keras import regularizers
import numpy as np
import utils

class NonlinearIB(Layer):              
    def __init__(self,
                 beta,                           
                 test_phase_noise      = False,
                 init_kde_logvar       = -5.,
                 init_noise_logvar     = -10.,
                 *kargs, **kwargs):
        """
        Construct a Keras layer for implementing nonlinear IB
        
        Parameters
        ----------
        beta : float
            beta parameter determines strength of IB penalization
        test_phase_noise : bool, optional
            Whether to add noise during the test phase, in addition to train phase
        init_kde_logvar : float, optional
            Initial (logarithm of) variance of KDE estimator
        init_noise_logvar : float, optional
            Initial (logarithm of) variance of Gaussian noise
        """
        
        self.supports_masking      = True
        self.uses_learning_phase   = True
        
        self.beta                  = beta
        
        self.init_kde_logvar       = init_kde_logvar
        self.init_noise_logvar     = init_noise_logvar
        self.test_phase_noise      = test_phase_noise
        
        super(NonlinearIB, self).__init__(*kargs, **kwargs)
        
        
    def build(self, input_shape):
        super(NonlinearIB, self).build(input_shape)
        self.beta_var     = self.add_weight((1,1), name='beta_var'   , trainable=False,
                                            initializer=lambda x: self.beta)
        self.kde_logvar   = self.add_weight((1,1), name='kde_logvar' , trainable=False,
                                            initializer=lambda x: self.init_kde_logvar)
        self.noise_logvar = self.add_weight((1,1), name='noise_logvar', trainable=False,
                                            initializer=lambda x: self.init_noise_logvar)
         
    def get_training_callbacks(self, 
                               model,
                               trn,
                               minibatchsize,
                               train_kde_logvar=True, 
                               train_noise_logvar=True): # TODO documentation
        cbs = []
        if train_kde_logvar:
            cbs.append( utils.ParameterTrainer(loss=self.kde_loo_loss, parameter=self.kde_logvar, trn=trn, minibatchsize=minibatchsize) )
        if train_noise_logvar:
            cbs.append( utils.ParameterTrainer(loss=model.total_loss, parameter=self.noise_logvar, trn=trn, minibatchsize=minibatchsize) )
            
        return cbs

    def call(self, x, training=None):
        dims = K.cast( K.shape(x)[1], K.floatx() ) 
        N    = K.cast( K.shape(x)[0], K.floatx() )

        # Gets an NxN matrix of pairwise distances between each vector in minibatch
        dists = utils.dist_mx(x)
        
        # Computes an estimate of the mutual information
        total_var   = K.exp(self.noise_logvar) + K.exp(self.kde_logvar)
        normconst   = (dims/2.0)*K.log(2*np.pi*total_var)
        lprobs      = utils.logsumexp(-dists / (2*total_var), axis=1) - K.log(N) - normconst
        h           = -K.mean(lprobs)
        hcond       = normconst
        self.mi     = h - hcond
        
        self.add_loss([K.in_train_phase(self.beta_var * self.mi, K.variable(0.0), training),])
        
        
        # Computes an estimate of the leave-one-out log probability loss, used
        # for training KDE bandwidth
        kdevar          = K.exp(self.kde_logvar)
        normconst_kde   = (dims/2.0)*K.log(2*np.pi*kdevar)
        scaleddists_kde = dists / (2*kdevar)

        scaleddists_kde += utils.tensor_eye(K.cast(N, 'int32')) * 10e20    # Dists should have very 
                                                                           # large values on diagonal 
                                                                           # (to make those contributions drop out)
        lprobs_kde = utils.logsumexp(-scaleddists_kde, axis=1) - K.log(N-1) - normconst_kde
        self.kde_loo_loss = -K.mean(lprobs_kde)

        # Now, simple add noise with appropriate variance
        noise = K.exp(0.5*self.noise_logvar) * K.random_normal(shape=K.shape(x))
        
        if self.test_phase_noise:
            return x + noise
        else:
            return K.in_train_phase(x + noise, x, training)
        
