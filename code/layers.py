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
                 *kargs, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        
        self.init_logvar = init_logvar
        self.logvar      = K.variable(0.0)
        
        self.logvar_trainable = logvar_trainable
        self.test_phase_noise = test_phase_noise
        
        super(NoiseLayer, self).__init__(*kargs, **kwargs)
        
    def build(self, input_shape):
        super(NoiseLayer, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        
        if self.logvar_trainable:
            self.trainable_weights = [self.logvar,]
        else:
            self.trainable_weights = []
        
    def get_noise(self, x):
        
        return K.exp(0.5*self.logvar) * K.random_normal(shape=K.shape(x), mean=0., stddev=1)
    
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
        

if K._BACKEND == 'tensorflow':
    def K_n_choose_k(n, k, seed=None):
        import tensorflow as tf
        if seed is None:
            seed = np.random.randint(10e6)
        x = tf.range(0, limit=n, dtype='int32')
        x = tf.random_shuffle(x, seed=seed)
        x = x[0:k]
        return x
    
else:
    def K_n_choose_k(n, k, seed=None):
        from theano.tensor.shared_randomstreams import RandomStreams
        if seed is None:
            seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        r = rng.choice(size=(k,), a=n, replace=False, dtype='int32')
        return r
    
class MICalculator(regularizers.Regularizer):
    def __init__(self, beta, model_layers, same_batch=False, data=None, miN=1000, init_kde_logvar=-5.):
        # miN is the batch size used to compute MI estimates
        self.beta            = beta
        self.init_kde_logvar = init_kde_logvar
        self.model_layers    = model_layers
        self.same_batch = same_batch
        
        self.miN  = miN
        self.set_data(data)
            
        # this should be constructed *before* NoiseLayer is added
        for layer in self.model_layers:
            if isinstance(layer, NoiseLayer):
                raise Exception("Model should not have NoiseLayer")
        
        self.kde_logvar = K.variable(self.init_kde_logvar)
        
        super(MICalculator, self).__init__()

    def set_data(self, data):
        self.data = data
        self._sample_noise_layer_input = None
        
    def set_noiselayer(self, noiselayer):
        self.noise_logvar = noiselayer.logvar
          
    @property
    def sample_noise_layer_input(self):
        if self._sample_noise_layer_input is None:
            if self.data is None:
                raise Exception("data attribute not initialized")
            if K._BACKEND == 'tensorflow':
                import tensorflow as tf
                c_input = tf.constant(self.data) 
            else:
                c_input = K.variable(self.data)
            input_ndxs = K_n_choose_k(len(self.data), self.miN)
            noise_layer_input = K.gather(c_input, input_ndxs)

            for layerndx, layer in enumerate(self.model_layers):
                noise_layer_input = layer.call(noise_layer_input)
            self._sample_noise_layer_input = noise_layer_input
                    
        return self._sample_noise_layer_input
    
    def noise_layer_input(self, x):
        if self.same_batch:
            return x
        else:
            return self.sample_noise_layer_input

    def get_h(self, x=None):
        # returns entropy
        current_var = K.exp(self.noise_logvar) + K.exp(self.kde_logvar)
        return kde_entropy(self.noise_layer_input(x), current_var)

    def get_hcond(self, x=None):
        # returns conditional entropy
        return kde_condentropy(self.noise_layer_input(x), K.exp(self.noise_logvar))

    def get_mi(self, x=None):
        mi = self.get_h(x) - self.get_hcond(x)
        return mi
    
    def __call__(self, x):
        return K.in_train_phase(self.beta * self.get_mi(x), K.variable(0.0))
