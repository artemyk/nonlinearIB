import keras.backend as K
from keras.layers import Layer
import numpy as np
from entropy import *


class NoiseLayer(Layer):
    # with variable noise
    def __init__(self, 
                 init_logvar    = -10.,
                 logvar_trainable = True,
                 test_phase_noise = True,
                 mi_calculator    = None, 
                 init_beta        = 0.0,
                 *kargs, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        
        self.init_logvar = init_logvar
        self.init_beta   = init_beta
        self.beta        = K.variable(0.0)
        self.logvar      = K.variable(0.0)
        
        self.logvar_trainable = logvar_trainable
        self.test_phase_noise = test_phase_noise

        self.mi_calculator = mi_calculator
        
        super(NoiseLayer, self).__init__(*kargs, **kwargs)
        
    def build(self, input_shape):
        super(NoiseLayer, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        K.set_value(self.beta, self.init_beta)
        
        if self.logvar_trainable:
            self.trainable_weights = [self.logvar,]
        else:
            self.trainable_weights = []

        if self.mi_calculator is not None:
            self.add_loss(K.in_train_phase(self.beta*self.mi_calculator.get_mi(self.logvar), K.variable(0.0)))
        
    def get_noise(self, x):
        return K.exp(0.5*self.logvar) * K.random_normal(shape=K.shape(x), mean=0., std=1)
    
    def call(self, x, mask=None):
        if self.test_phase_noise:
            return x+self.get_noise(x)
        else:
            return K.in_train_phase(x+self.get_noise(x), x)


class MICalculator(object):
    def __init__(self, model_layers, input_samples, init_kde_logvar=-5.):
        self.init_kde_logvar = init_kde_logvar
        self.input_samples   = input_samples
        self.model_layers    = model_layers

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
        self.kde_logvar = K.variable(self.init_kde_logvar)

    def get_h(self, noise_logvar):
        # returns entropy
        current_var = K.exp(noise_logvar) + K.exp(self.kde_logvar)
        return kde_entropy(self.noise_layer_input, current_var)

    def get_hcond(self, noise_logvar):
        # returns entropy
        return kde_condentropy(self.noise_layer_input, K.exp(noise_logvar))

    def get_mi(self, noise_logvar):
        return self.get_h(noise_logvar) - self.get_hcond(noise_logvar)

