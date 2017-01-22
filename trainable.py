import keras.backend as K
from keras.layers import Layer
from keras.callbacks import Callback
import numpy as np
import scipy.optimize
from entropy import *


class NoiseLayer(Layer):
    # with variable noise
    def __init__(self, 
                 init_logvar    = -10.,
                 logvar_trainable = True,
                 test_phase_noise = True,
                 mi_calculator    = None, 
                 init_alpha       = 0.0,
                 *kargs, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        
        self.init_logvar = init_logvar
        self.init_alpha  = init_alpha
        self.alpha       = K.variable(0.0)
        self.logvar      = K.variable(0.0)
        
        self.logvar_trainable = logvar_trainable
        self.test_phase_noise = test_phase_noise

        self.mi_calculator = mi_calculator
        
        super(NoiseLayer, self).__init__(*kargs, **kwargs)
        
    def build(self, input_shape):
        super(NoiseLayer, self).build(input_shape)
        K.set_value(self.logvar, self.init_logvar)
        K.set_value(self.alpha, self.init_alpha)
        
        if self.logvar_trainable:
            self.trainable_weights = [self.logvar,]
        else:
            self.trainable_weights = []

        if self.mi_calculator is not None:
            self.add_loss(K.in_train_phase(self.alpha*self.mi_calculator.get_mi(self.logvar), K.variable(0.0)))
        
    def get_noise(self, x):
        return K.exp(0.5*self.logvar) * K.random_normal(shape=K.shape(x), mean=0., std=1)
    
    def call(self, x, mask=None):
        if self.test_phase_noise:
            return x+self.get_noise(x)
        else:
            return K.in_train_phase(x+self.get_noise(x), x)


# class NoiseTrain(Callback):
#     def __init__(self, traindata, noiselayer):
#         super(NoiseTrain, self).__init__()
#         self.traindata = traindata
#         self.noiselayer = noiselayer
        
#     def on_train_begin(self, logs={}):
#         modelobj = self.model.model
#         inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
#         lossfunc = K.function(inputs, [modelobj.total_loss])
#         jacfunc  = K.function(inputs, K.gradients(modelobj.total_loss, self.noiselayer.logvar))
#         sampleweights = np.ones(len(self.traindata.X))
#         def obj(logvar):
#             v = K.get_value(self.noiselayer.logvar)
#             K.set_value(self.noiselayer.logvar, logvar.flat[0])
#             r = lossfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])[0]
#             K.set_value(self.noiselayer.logvar, v)
#             return r
#         def jac(logvar):
#             v = K.get_value(self.noiselayer.logvar)
#             K.set_value(self.noiselayer.logvar, logvar.flat[0])
#             r = np.atleast_2d(np.array(jacfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])))[0]
#             K.set_value(self.noiselayer.logvar, v)
#             return r
            
#         self.obj = obj
#         self.jac = jac
        
#     def on_epoch_begin(self, epoch, logs={}):
#         r = scipy.optimize.minimize(self.obj, K.get_value(self.noiselayer.logvar), jac=self.jac)
#         best_val = r.x[0]
#         cval =  K.get_value(self.noiselayer.logvar)
#         max_var = 1.0 + cval
#         if best_val > max_var:
#             # don't raise it too fast, so that gradient information is preserved 
#             best_val = max_var
            
#         K.set_value(self.noiselayer.logvar, best_val)

        

class KDETrain(Callback):
    def __init__(self, mi_calculator, *kargs, **kwargs):
        super(KDETrain, self).__init__(*kargs, **kwargs)
        self.mi_calculator = mi_calculator
        
    def on_train_begin(self, logs={}):
        #self.nlayerinput = lambda x: K.function([self.model.layers[0].input], [self.kdelayer.input])([x])[0]
        N, dims = self.mi_calculator.input_samples.shape
        Kdists = K.placeholder(ndim=2)
        Klogvar = K.placeholder(ndim=0)
            
        lossfunc = K.function([Kdists, Klogvar,], [kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar))])
        jacfunc  = K.function([Kdists, Klogvar,], K.gradients(kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar)), Klogvar))

        def obj(logvar, dists):
            return lossfunc([dists, logvar.flat[0]])[0]
        def jac(logvar, dists):
            return np.atleast_2d(np.array(jacfunc([dists, logvar.flat[0]])))[0] 

        self.obj = obj
        self.jac = jac

    @staticmethod
    def get_dists(output):
        N, dims = output.shape

        # Kernel density estimation of entropy
        y1 = output[None,:,:]
        y2 = output[:,None,:]

        dists = np.sum((y1-y2)**2, axis=2) 
        return dists
    
    def on_epoch_begin(self, epoch, logs={}):
        vals = K.eval(self.mi_calculator.noise_layer_input)
        dists = self.get_dists(vals)
        dists += 10e20 * np.eye(dists.shape[0])
        r = scipy.optimize.minimize(self.obj, K.get_value(self.mi_calculator.kde_logvar).flat[0], 
                                    jac=self.jac, 
                                    args=(dists,),
                                    )
        best_val = r.x.flat[0]
        K.set_value(self.mi_calculator.kde_logvar, best_val)

 
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

