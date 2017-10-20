import keras.backend as K
from keras.layers import Layer
from keras import regularizers
import numpy as np
import utils

class NonlinearIB(Layer):  # NonlinearIB layer
    def __init__(self,
                 beta,
                 noise_level_trainable = True,
                 test_phase_noise      = False,
                 init_kde_logvar       = -5.,
                 init_noise_logvar     = -10.,
                 mi_minibatchsize      = None,
                 input_data            = None, 
                 *kargs, **kwargs):
        
        self.supports_masking      = True
        self.uses_learning_phase   = True
        
        self.beta                  = beta
        
        self.init_noise_logvar     = init_noise_logvar
        self.noise_logvar          = K.variable(0.0, name='noise_logvar')
        self.noise_level_trainable = noise_level_trainable
        self.test_phase_noise      = test_phase_noise
        
        self.init_kde_logvar       = init_kde_logvar
        self.kde_logvar            = K.variable(0.0, name='kde_logvar')
        
        if input_data is None and mi_minibatchsize is not None:
            raise Exception('mi_minibatchsize is specified but input_data is not provided')
        if input_data is not None and mi_minibatchsize is None:
            raise Exception('input_data provided by mi_minibatchsize not specified')
        self.mi_minibatchsize      = mi_minibatchsize
        self.input_data            = input_data
        
        super(NonlinearIB, self).__init__(*kargs, **kwargs)
        
        
    def build(self, input_shape):
        super(NonlinearIB, self).build(input_shape)
        K.set_value(self.noise_logvar, self.init_noise_logvar)
        K.set_value(self.kde_logvar  , self.init_kde_logvar)
        
        self.trainable_weights = []
        self.trainable_weights.append(self.kde_logvar)
        if self.noise_level_trainable:
            self.trainable_weights.append(self.noise_logvar)
            

    def get_mi_minibatch(self, x):
        if self.mi_minibatchsize is None:
            return x
        else:
            input_ndxs = utils.K_n_choose_k(len(self.input_data), self.mi_minibatchsize)
            input_sample = K.gather(utils.tensor_constant(self.input_data), input_ndxs)

            def get_layer_output(clayer, baseinputdata):
                if clayer is None:
                    return baseinputdata
                else:
                    return clayer.call(get_layer_output(utils.get_input_layer(clayer), baseinputdata))

            if hasattr(x, 'len'):
                if len(x) > 1:
                    raise Exception('NonlinearIB only takes a single input layer')
                inbound_layer = x[0]._keras_history[0]
            else:
                inbound_layer = x._keras_history[0]
                
            return get_layer_output(inbound_layer, input_sample)
        
    
    def IBpenalty(self, minibatch):        
        dims = K.cast( K.shape(minibatch)[1], K.floatx() ) 
        N    = K.cast( K.shape(minibatch)[0], K.floatx() )

        # get dists matrix
        dists = utils.dist_mx(minibatch)
        
        # overall mutual informaiton
        total_var   = K.exp(self.noise_logvar) + K.exp(K.stop_gradient(self.kde_logvar))
        normconst   = (dims/2.0)*K.log(2*np.pi*total_var)
        lprobs      = utils.logsumexp(- dists / (2*total_var), axis=1) - K.log(N) - normconst
        h           = -K.mean(lprobs)
        hcond       = normconst
        mi          = h - hcond

        # leave one out entropy
        kdevar          = K.exp(self.kde_logvar)
        normconst_kde   = (dims/2.0)*K.log(2*np.pi*kdevar)
        # Dists should have very large values on diagonal (to make those contributions drop out)
        scaleddists_kde = K.stop_gradient(dists) / (2*kdevar)
        scaleddists_kde += utils.tensor_eye(K.cast(N, 'int32')) * 10e20
        lprobs_kde = utils.logsumexp(-scaleddists_kde, axis=1) - K.log(N-1) - normconst_kde
        hloo = -K.mean(lprobs_kde)
        
        return [mi, hloo]
        loss = K.in_train_phase(self.beta * mi + hloo, K.variable(0.0), training)
        return loss
    
    def call(self, x, training=None):
        minibatch = self.get_mi_minibatch(x)
        mi, hloo  = self.IBpenalty(minibatch)
        self.add_loss([K.in_train_phase(self.beta * mi + hloo, K.variable(0.0), training),])
        
        noise = K.exp(0.5*self.noise_logvar) * K.random_normal(shape=K.shape(x))
        
        if self.test_phase_noise:
            return x + noise
        else:
            return K.in_train_phase(x + noise, x, training)
        
    def get_config(self):
        config = {'kde_logvar': self.kde_logvar, 'noise_logvar': self.noise_logvar}
        base_config = super(NonlinearIB, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

