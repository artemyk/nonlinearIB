# Implementation of 
# A Kolchinsky, BD Tracey, DH Wolpert, "Nonlinear Information Bottleneck", https://arxiv.org/abs/1705.02436
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer
import keras
import numpy as np
import utils

LOG2PI = np.log(2*np.pi, dtype=K.floatx())

class NonlinearIB(Layer):              
    def __init__(self,
                 beta,                           
                 test_phase_noise      = False,
                 init_kde_logvar       = -5.,
                 init_noise_logvar     = -10.,
                 noise_logvar_train_firstepoch = 0,
                 trainable_noise_logvar = True,
                 trainable_kde_logvar   = True,
                 deterministic_ib       = False,
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
        noise_logvar_train_firstepoch : int, optional
            First epoch on which to begin training noise_logvar.  None to 
            eliminate training of this parameter
        """
        
        self.supports_masking      = True
        self.uses_learning_phase   = True
        
        self.beta                  = beta
        
        self.init_kde_logvar       = init_kde_logvar
        self.init_noise_logvar     = init_noise_logvar
        self.test_phase_noise      = test_phase_noise
        #self.noise_logvar_train_firstepoch = noise_logvar_train_firstepoch
        
        self.trainable_noise_logvar = trainable_noise_logvar
        self.trainable_kde_logvar   = trainable_kde_logvar
        
        self.deterministic_ib       = deterministic_ib
        
        super(NonlinearIB, self).__init__(*kargs, **kwargs)
        
        
    def build(self, input_shape):
        super(NonlinearIB, self).build(input_shape)
        self.beta_var     = self.add_weight((1,1), name='beta_var'   , trainable=False,
                                            initializer=lambda x: self.beta)
        self.kde_logvar   = self.add_weight((1,1), name='kde_logvar' , trainable=False,
                                            initializer=lambda x: self.init_kde_logvar)
        self.noise_logvar = self.add_weight((1,1), name='noise_logvar', trainable=self.trainable_noise_logvar,
                                            initializer=lambda x: self.init_noise_logvar)
        #self.include_mi_loss = self.add_weight((1,1), name='include_mi_loss', trainable=False,
        #                                       initializer=lambda x: 0.0)
         
    def get_training_callbacks(self, 
                               model,
                               trn,
                               minibatchsize):
                               #train_kde_logvar=True, 
                               #train_noise_logvar=True): # TODO documentation
        cbs = []
        if self.trainable_kde_logvar:
            cbs.append( utils.ParameterTrainer(loss=self.kde_loo_loss, parameter=self.kde_logvar, trn=trn, minibatchsize=minibatchsize) )
            
#         def turn_on_mi_loss(epoch, logs):
#             if self.noise_logvar_train_firstepoch == epoch:
#                 K.set_value(self.include_mi_loss, 1.0)
            
#         cbs.append(keras.callbacks.LambdaCallback(on_epoch_begin=turn_on_mi_loss,
#                                                   on_train_begin=lambda logs: K.set_value(self.include_mi_loss, 0.0)))
                   
#         if train_noise_logvar:
#             cbs.append( utils.ParameterTrainer(loss=model.total_loss, parameter=self.noise_logvar, trn=trn, minibatchsize=minibatchsize,
#                                                first_epoch=self.noise_logvar_train_firstepoch) )
#                        #,
#                        #                       init_value=-1.), bounds=[[-10.,5.],]) ) # TODO would be nice to eliminate this stuff
            
        return cbs

    def get_ib_cost(self, x, dists=None):
        # Computes an estimate of the mutual information
        # pass in distance matrix if it already exists, so it doesn't have to be recomputed
        dims = K.cast( K.shape(x)[1], K.floatx() ) 
        N    = K.cast( K.shape(x)[0], K.floatx() )
        
        if dists is None:
            dists = utils.dist_mx(x)
        
        # total_logvar= K.logsumexp([self.noise_logvar, self.kde_logvar]) # K.exp(self.noise_logvar) + K.exp(self.kde_logvar)
        # TODO total_logvar= K.clip(total_logvar, 0, 85) # avoid numerical overflows
        #normconst   = (dims/2.0)*(LOG2PI + total_logvar)
        #lprobs      = K.logsumexp(-dists / (2*K.exp(total_logvar)), axis=1) - K.log(N) - normconst
        total_var   = K.exp(self.noise_logvar) + K.exp(self.kde_logvar)
        normconst   = (dims/2.0)*(LOG2PI + K.log(total_var))
        lprobs      = K.logsumexp(-dists / (2*total_var), axis=1) - K.log(N) - normconst
        h           = -K.mean(lprobs)
        
        if self.deterministic_ib:
            return h
        else:
            hcond  = (dims/2.0)*(LOG2PI + self.noise_logvar)
            return h - hcond
        
        
    def call(self, x, training=None):
        dims = K.cast( K.shape(x)[1], K.floatx() ) 
        N    = K.cast( K.shape(x)[0], K.floatx() )

        # Gets an NxN matrix of pairwise distances between each vector in minibatch
        dists = utils.dist_mx(x)
        
        self.mi = self.get_ib_cost(x, dists)
        
        # TODO
        #self.add_loss([K.in_train_phase(self.include_mi_loss * self.beta_var * self.mi, K.variable(0.0), training),])
        #self.add_loss([K.in_train_phase(self.beta_var * self.mi, K.variable(0.0), training),])
        self.add_loss([self.beta_var * self.mi,])
        
        # Computes an estimate of the leave-one-out log probability loss, used
        # for training KDE bandwidth
        kdevar          = K.exp(self.kde_logvar)
        normconst_kde   = (dims/2.0)*K.log(2*np.pi*kdevar)
        scaleddists_kde = dists / (2*kdevar)

        scaleddists_kde += utils.tensor_eye(K.cast(N, 'int32')) * 10e20    # Dists should have very 
                                                                           # large values on diagonal 
                                                                           # (to make those contributions drop out)
        lprobs_kde = K.logsumexp(-scaleddists_kde, axis=1) - K.log(N-1) - normconst_kde
        self.kde_loo_loss = -K.mean(lprobs_kde)

        #x = K.tile(x, [2,1])

        # Now, simple add noise with appropriate variance
        noise = K.exp(0.5*self.noise_logvar) * K.random_normal(shape=K.shape(x))
        
        x_plus_noise = x + noise
        if self.test_phase_noise:
            return x_plus_noise
        else:
            return K.in_train_phase(x_plus_noise, x, training)
        

    #def compute_output_shape(self, input_shape):
    #    return (2000, input_shape[1])