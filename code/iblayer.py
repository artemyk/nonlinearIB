import numpy as np
import tensorflow as tf
import entropy
import utils

class NoisyIBLayer(tf.keras.layers.Layer):
    def __init__(self, init_noisevar=0.01, **kwargs):
        # init_noisevar    : initial noise variance
        
        super(NoisyIBLayer, self).__init__(**kwargs)
        init_phi           = utils.softplusinverse(init_noisevar).astype('float32') # softplus inverse
        
        # phi is the noise variance (in softplus space) 
        self.phi      = tf.get_variable('phi', dtype=tf.float32, trainable=True, initializer=init_phi)
        self.noisevar = tf.nn.softplus(self.phi)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = int(input_shape[1])
        self.H_T_given_X = entropy.gaussian_entropy(self.input_dim, self.noisevar)
        
        self.input_spec = tf.layers.InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True
        
    def call(self, inputs):
        self.T_nonoise     = tf.identity(inputs, name='RawInput') # useful to name Tensor for finding later
        
        self.dist_matrix   = entropy.pairwise_distance(inputs)
        
        self.H_T_lb        = entropy.GMM_entropy(self.dist_matrix, self.noisevar, self.input_dim, 'lower')
        self.Ixt_lb        = self.H_T_lb - self.H_T_given_X
        
        self.H_T           = entropy.GMM_entropy(self.dist_matrix, self.noisevar, self.input_dim, 'upper')
        self.Ixt           = self.H_T - self.H_T_given_X
        
        # MI as calculated by Variational IB estimator, based on Alemi
        # https://github.com/alexalemi/vib_demo/blob/master/MNISTVIB.ipynb
        tfd                = tf.contrib.distributions
        prior              = tfd.Normal(0.0, 1.0)
        encoding           = tfd.Normal(inputs, self.noisevar)
        self.vIxt          = tf.reduce_sum(tf.reduce_mean(tfd.kl_divergence(encoding, prior), 0))

        self.noise         = tf.random_normal(shape=tf.shape(inputs), mean=0.0, stddev=1, dtype=tf.float32)
                
        self.input_layer   = inputs
        
        return inputs + self.noise * tf.sqrt(self.noisevar)
   