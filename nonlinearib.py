# Implementation of 
# A Kolchinsky, BD Tracey, DH Wolpert, "Nonlinear Information Bottleneck", https://arxiv.org/abs/1705.02436
from __future__ import print_function

import keras.backend as K
from keras.layers import Layer
from keras import regularizers
import numpy as np
import utils

def sample_minibatch(data, n):
    ndxs = utils.K_n_choose_k(len(data), n)
    sample = K.gather(utils.tensor_constant(data), ndxs)
    return sample
    
class NonlinearIB(Layer):              
    def __init__(self,
                 beta,                           
                 kde_trainable         = True,   # 
                 noise_level_trainable = True,
                 test_phase_noise      = False,
                 init_kde_logvar       = -5.,
                 init_noise_logvar     = -10.,
                 mi_minibatchsize      = None,
                 input_data            = None, 
                 *kargs, **kwargs):
        """
        Construct a Keras layer for implementing nonlinear IB
        
        Parameters
        ----------
        beta : float
            beta parameter determines strength of IB penalization
        init_kde_logvar : float, optional
            Initial (logarithm of) variance of KDE estimator
        init_noise_logvar : float, optional
            Initial (logarithm of) variance of Gaussian noise
        """

        # Previously, we could also train the kde_logvar and noise_logvar through
        # gradient descent.  However, it seems to work better to set these to
        # their optimal values once an epoch, via callback.
        #
        # Hence, ignore this code below.
        
        """
        kde_trainable : bool, optional
            Whether to optimize leave-one-out likelihood w.r.t. KDE variance 
        noise_level_trainable : bool, optional
            Whether to optimize IB functional w.r.t. the level of injected 
            Gaussian noise
        mi_minibatchsize : int, optional
            Number of samples to include in SGD minibatches for estimating 
            MI(hidden;input) and leave-one-out likelihood terms. If None,
            then regular SGD batches are used
        input_data : np.array, optional
            If mi_minibatchsize is not None, then this should be the input
            training data, from which minibatches are drawn
        """
        
        self.noise_level_trainable = noise_level_trainable
        self.kde_trainable         = kde_trainable
        if input_data is None and mi_minibatchsize is not None:
            raise Exception('mi_minibatchsize is specified but input_data is not provided')
        if input_data is not None and mi_minibatchsize is None:
            raise Exception('input_data provided by mi_minibatchsize not specified')
        self.mi_minibatchsize      = mi_minibatchsize
        self.input_data            = input_data
        
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
        self.kde_logvar   = self.add_weight((1,1), name='kde_logvar' , trainable=False, # self.kde_trainable,
                                            initializer=lambda x: self.init_kde_logvar)
        self.noise_logvar = self.add_weight((1,1), name='noise_logvar', trainable=False, # self.noise_level_trainable,
                                            initializer=lambda x: self.init_noise_logvar)


    def init_input_to_me_map(self, x):
        # TODO describe x
        # TODO describe this function, whta it returns
        # To generate new minimatches, we have to manually propagate samples of input
        # data through all the layers that preceeded this layer.  

        # The following code the incoming layer to this one (note that inbound_nodes
        # might not be defined yet, so we extract it from _keras_history
        if not hasattr(self, '_input_to_me_map'):
            if hasattr(x, 'len'):
                if len(x) > 1:
                    raise Exception('NonlinearIB only takes a single-input layesr')
                inbound_layer = x[0]._keras_history[0]
            else:
                inbound_layer = x._keras_history[0]

            # We then backtrack from this layer through earlier and earlier layers,
            # recursively propagating the input data
            layer_calls = []
            def traceback_layer(clayer):
                if clayer is None:
                    return
                else:
                    layer_calls.append(clayer.call)
                    traceback_layer(utils.get_input_layer(clayer))

            traceback_layer(inbound_layer)
            
            f = layer_calls[-1]
            for c in layer_calls[:-1:-1]:
                f = c(f)
            self.input_to_me_map = f
        
                

#    @property
#    def minibatch(self):
#        return self.input_to_current_map(sample_minibatch(self.input_data, 
#     def get_mi_minibatch(self, x):
#         if self.mi_minibatchsize is None:
#             # If self.mi_minibatchsize is None, then just use regular SGD batches
#             # for estimating information-theoretic terms
#             return x
#         else:
#             if not hasattr(self, '_minibatch_var'):
#                 # TODO document
                
#                 # Sample a random minibatch from self.input_data
#                 input_ndxs = utils.K_n_choose_k(len(self.input_data), self.mi_minibatchsize)
#                 input_sample = K.gather(utils.tensor_constant(self.input_data), input_ndxs)

#                 # To generate new minimatches, we have to manually propagate samples of input
#                 # data through all the layers that preceeded this layer.  

#                 # The following code the incoming layer to this one (note that inbound_nodes
#                 # might not be defined yet, so we extract it from _keras_history

#                 if hasattr(x, 'len'):
#                     if len(x) > 1:
#                         raise Exception('NonlinearIB only takes a single input layer')
#                     inbound_layer = x[0]._keras_history[0]
#                 else:
#                     inbound_layer = x._keras_history[0]

#                 # We then backtrack from this layer through earlier and earlier layers,
#                 # recursively propagating the input data 
#                 def get_layer_output(clayer, baseinputdata):
#                     if clayer is None:
#                         return baseinputdata
#                     else:
#                         return clayer.call(get_layer_output(utils.get_input_layer(clayer), baseinputdata))

#                 self._minibatch_var = get_layer_output(inbound_layer, input_sample)
            
#             return self._minibatch_var
        
    
    def IBpenalty(self, minibatch):        
        dims = K.cast( K.shape(minibatch)[1], K.floatx() ) 
        N    = K.cast( K.shape(minibatch)[0], K.floatx() )

        # Gets an NxN matrix of pairwise distances between each vector in minibatch
        dists = utils.dist_mx(minibatch)
        

        #cLV = K.stop_gradient(2*K.log((4./(dims+2.))**(1./(dims+4.))*N**(-1.0/(dims+4.))*K.std(K.stop_gradient(minibatch))))
        #print('estimated logvar',K.eval(cLV))
        # Computes an estimate of the mutual information
        # Note that we don't want to optimize self.kde_logvar so as to minimize this
        # mutual information 
        total_var   = K.exp(self.noise_logvar) + K.exp(K.stop_gradient(self.kde_logvar)) # TODO
        #total_var   = K.exp(self.noise_logvar) + K.exp(cLV) # TODO
        normconst   = (dims/2.0)*K.log(2*np.pi*total_var)
        lprobs      = utils.logsumexp(-dists / (2*total_var), axis=1) - K.log(N) - normconst
        h           = -K.mean(lprobs)
        hcond       = normconst
        mi          = h - hcond

    
    def get_kde_loo_loss(self, x):
        # leave one out entropy
        dims = K.cast( K.shape(x)[1], K.floatx() ) 
        N    = K.cast( K.shape(x)[0], K.floatx() )

        dists = utils.dist_mx(x)
        
        kdevar          = K.exp(self.kde_logvar)
        normconst_kde   = (dims/2.0)*K.log(2*np.pi*kdevar)
        scaleddists_kde = dists / (2*kdevar)

        # Dists should have very large values on diagonal (to make those contributions drop out)
        scaleddists_kde += utils.tensor_eye(K.cast(N, 'int32')) * 10e20
        lprobs_kde = utils.logsumexp(-scaleddists_kde, axis=1) - K.log(N-1) - normconst_kde
        return -K.mean(lprobs_kde)
         
    
    def call(self, x, training=None):
        self.init_input_to_me_map(x)
        #minibatch = self.get_mi_minibatch(x)
        #mi, hloo  = self.IBpenalty(minibatch)
        
        dims = K.cast( K.shape(x)[1], K.floatx() ) 
        N    = K.cast( K.shape(x)[0], K.floatx() )

        # Gets an NxN matrix of pairwise distances between each vector in minibatch
        dists = utils.dist_mx(x)
        
        # Computes an estimate of the mutual information
        # mutual information 
        total_var   = K.exp(self.noise_logvar) + K.exp(K.stop_gradient(self.kde_logvar)) # TODO
        normconst   = (dims/2.0)*K.log(2*np.pi*total_var)
        lprobs      = utils.logsumexp(-dists / (2*total_var), axis=1) - K.log(N) - normconst
        h           = -K.mean(lprobs)
        hcond       = normconst
        mi          = h - hcond
        
        self.add_loss([K.in_train_phase(self.beta_var * mi, K.variable(0.0), training),])
        #self.add_loss([K.in_train_phase(self.beta_var * mi + hloo, K.variable(0.0), training),])
        
        self.kde_loo_loss = self.get_kde_loo_loss(x)
        
        noise = K.exp(0.5*self.noise_logvar) * K.random_normal(shape=K.shape(x))
        
        if self.test_phase_noise:
            return x + noise
        else:
            return K.in_train_phase(x + noise, x, training)
        
#     def get_config(self):
#         config = {'beta': self.beta_var, 'kde_logvar': self.kde_logvar, 'noise_logvar': self.noise_logvar}
#         base_config = super(NonlinearIB, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))

