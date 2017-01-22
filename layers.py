import keras.backend as K
from keras.layers import Layer
from entropy import kde_entropy, kde_condentropy
import tensorflow as tf

# class MIPenaltyLayer(Layer):
#     # with variable noise
#     def __init__(self, 
#                  input_layer,
#                  noise_layer,
#                  init_kde_logvar      = -5., 
#                  mi_samples           = None, 
#                  init_alpha           = 1.0,
#                  *kargs, **kwargs):
#         self.supports_masking = True
#         self.uses_learning_phase = True
        
#         self.init_alpha        = init_alpha
#         self.init_kde_logvar   = init_kde_logvar
#         self.alpha             = K.variable(0.0)
#         self.kde_logvar        = K.variable(0.0)

#         self.mi_samples        = mi_samples
#         self.noise_layer = noise_layer
#         self.input_layer = input_layer
        
#         super(MIPenaltyLayer, self).__init__(*kargs, **kwargs)
        
#     def build(self, input_shape):
#         super(MIPenaltyLayer, self).build(input_shape)
#         K.set_value(self.kde_logvar, self.init_kde_logvar)
#         K.set_value(self.alpha, self.init_alpha)

#         d = tf.constant(pd)
#         for layerndx, layer in enumerate(model.layers):
#             d = layer(d)
#         return d

            
#         current_var   = K.exp(self.noise_layer.logvar) + K.exp(self.kde_logvar)
#         current_input = K.function([self.input_layer.input], [self.noise_layer.input])([self.mi_samples])[0]
#         print current_input
#         mi = kde_entropy(current_input, current_var) - kde_condentropy(current_input, K.exp(self.noiselayer.logvar))
#         self.add_loss(K.in_train_phase(self.alpha * mi, K.variable(0.0)))
    
#     def call(self, x, mask=None):
#         return x


class NoiseLayer(Layer):
    # with variable noise
    def __init__(self, 
                 init_logvar    = -10.,
                 logvar_trainable = True,
                 test_phase_noise = True,
                 mi_calculator    = None, 
                 *kargs, **kwargs):
        self.supports_masking = True

        if test_phase_noise:
            self.uses_learning_phase = True
        
        self.init_logvar = init_logvar
        self.alpha       = K.variable(0.0)
        self.logvar      = K.variable(0.0)
        
        self.logvar_trainable = logvar_trainable
        self.test_phase_noise = test_phase_noise

        self.mi_calculator = mi_calculator
        
        super(NoiseLayer, self).__init__(*kargs, **kwargs)
        
    def build(self, input_shape):
        super(NoiseLayer, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        
        if self.logvar_trainable:
            self.trainable_weights = [self.logvar,]
        else:
            self.trainable_weights = []

        if self.mi_calculator is not None:
            self.mi_calculator.set_noise_layer(self)
            self.add_loss(self.mi_calculator.get_mi())
        
    def get_noise(self, x):
        return K.exp(0.5*self.logvar) * K.random_normal(shape=K.shape(x), mean=0., std=1)
    
    def call(self, x, mask=None):
        if self.test_phase_noise:
            return x+self.get_noise(x)
        else:
            return K.in_train_phase(x+self.get_noise(x), x)

class MICalculator(object):
    def __init__(self, model, input_samples, init_kde_logvar=-5., init_alpha=0.0):
        self.init_kde_logvar = init_kde_logvar
        self.init_alpha      = init_alpha

        # # Last layer should be NoiseLayer
        # self.noise_layer = model.layers[-1]
        # if not isinstance(self.noise_layer, NoiseLayer):
        #     raise Exception("Last layer of model should be NoiseLayer")

        import tensorflow as tf
        noise_layer_input = tf.constant(input_samples)
        for layerndx, layer in enumerate(model.layers):
            noise_layer_input = layer(noise_layer_input)
        self.noise_layer_input = noise_layer_input

    def set_noise_layer(self, noise_layer):
        self.noise_layer = noise_layer
        self.alpha = K.variable(self.init_alpha)
        self.kde_logvar = K.variable(self.init_kde_logvar)


    def get_mi(self):
        if self.noise_layer is None:
            raise Exception('Need to initialize noise_layer attribute')

        current_var   = K.exp(self.noise_layer.logvar) + K.exp(self.kde_logvar)
        h = kde_entropy(self.noise_layer_input, current_var)
        hcond = kde_condentropy(self.noise_layer_input, K.exp(self.noise_layer.logvar))
        mi = h - hcond
        return self.alpha * mi
#        self.add_loss(K.in_train_phase(self.alpha * mi, K.variable(0.0)))


        



# class KDEParamLayer(Layer):
#     # This layer k
#     # with variable noise
#     def __init__(self, init_logvar):
#         self.init_kde_logvar = init_kde_logvar
#         self.kde_logvar = K.variable(0.0)
#         super(KDEParamLayer, self).__init__()
        
#     def build(self, input_shape):
#         super(KDEParamLayer, self).build(input_shape)
#         self.trainable_weights = []

#     def call(self, x, mask=None):
#         return x
