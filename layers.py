import keras.backend as K
from keras.layers import Layer
from keras import regularizers
import numpy as np
from entropy import *


class NoiseLayer(Layer):
    # with variable noise
    def __init__(self, 
                 init_logvar    = -10.,
                 logvar_trainable = True,
                 test_phase_noise = False,
                 #activity_regularizer=None,
                 *kargs, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        
        self.init_logvar = init_logvar
        self.logvar      = K.variable(0.0)
        
        self.logvar_trainable = logvar_trainable
        self.test_phase_noise = test_phase_noise
        
        #self.activity_regularizer = regularizers.get(activity_regularizer)

        super(NoiseLayer, self).__init__(*kargs, **kwargs)
        
    def build(self, input_shape):
        super(NoiseLayer, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        
        if self.logvar_trainable:
            self.trainable_weights = [self.logvar,]
        else:
            self.trainable_weights = []
        
    def get_noise(self, x):
        return K.exp(0.5*self.logvar) * K.random_normal(shape=K.shape(x), mean=0., std=1)
    
    def call(self, x, mask=None):
        if self.test_phase_noise:
            return x+self.get_noise(x)
        else:
            return K.in_train_phase(x+self.get_noise(x), x)

class IdentityMap(Layer):
    def __init__(self, activity_regularizer=None, *kargs, **kwargs):
        self.supports_masking = True
        self.activity_regularizer = regularizers.get(activity_regularizer)
        super(IdentityMap, self).__init__(*kargs, **kwargs)
        
    def call(self, x, mask=None):
        return x
        
class NoiseLayerVIB(Layer):
    # with variable noise
    def __init__(self, 
                 mean_dims,
                 test_phase_noise = False,
                 *kargs, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        
        self.test_phase_noise = test_phase_noise
        self.mean_dims = mean_dims
        
        super(NoiseLayerVIB, self).__init__(*kargs, **kwargs)
        
    def get_noise(self, clogvars):
        cvars = K.exp(0.5*clogvars)
        return cvars * K.random_normal(shape=K.shape(clogvars), mean=0., std=1)
    
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.mean_dims)
    
    def call(self, x, mask=None):
        means, clogvars = x[:,:self.mean_dims], x[:,self.mean_dims:]
        with_noise = means + self.get_noise(clogvars)
        if self.test_phase_noise:
            return with_noise
        else:
            return K.in_train_phase(with_noise, means) 

class MIRegularizerBase(regularizers.Regularizer):
    def __call__(self, x):
        return K.in_train_phase(self.beta * self.get_mi(x), K.variable(0.0))

class MICalculatorVIB(MIRegularizerBase):
    def __init__(self, beta):
        self.beta = beta
        super(MICalculatorVIB, self).__init__()
        
    def set_noiselayer(self, noiselayer):
        self.noiselayer = noiselayer
        
    def get_mi(self, x):
        # 0.5 * [tr(Sigma) + ||u_1||^2 - k - ln ( |Sigma 0| )]
        dims = self.noiselayer.mean_dims
        means, logcovs = x[:,:dims], x[:,dims:]
        norms = K.square(means)
        norms = K.sum(norms, axis=1)
        #v = 0.5 * (dims * K.exp(self.noise_logvar) + norms - dims - dims*self.noise_logvar)
        v = 0.5*(K.sum(K.exp(logcovs), axis=1) + norms - float(dims) - K.sum(logcovs, axis=1))
        kl = nats2bits * K.mean(v)
        return kl
        
    
class MICalculator(MIRegularizerBase):
    def __init__(self, beta, model_layers, input_samples, init_kde_logvar=-5.):
        self.beta            = beta
        self.init_kde_logvar = init_kde_logvar
        self.model_layers    = model_layers

        self.kde_logvar = K.variable(self.init_kde_logvar)
        
        self.set_input_samples(input_samples)
        
        super(MICalculator, self).__init__()

    def set_input_samples(self, input_samples):
        self.input_samples = input_samples
        if K._BACKEND == 'tensorflow':
            import tensorflow as tf
            noise_layer_input = tf.constant(input_samples) 
        else:
            noise_layer_input = K.variable(input_samples)

        for layerndx, layer in enumerate(self.model_layers):
            if isinstance(layer, NoiseLayer):
                # this should be constructed *before* NoiseLayer is added
                raise Exception("Model should not have NoiseLayer")
            noise_layer_input = layer(noise_layer_input)
        self.noise_layer_input = noise_layer_input
        

    def set_noiselayer(self, noiselayer):
        self.noise_logvar = noiselayer.logvar
        
    def get_h(self, x=None):
        # returns entropy
        current_var = K.exp(self.noise_logvar) + K.exp(self.kde_logvar)
        return kde_entropy(self.noise_layer_input, current_var)

    def get_hcond(self, x=None):
        # returns entropy
        return kde_condentropy(self.noise_layer_input, K.exp(self.noise_logvar))

    def get_mi(self, x=None):
        return self.get_h() - self.get_hcond()
    
