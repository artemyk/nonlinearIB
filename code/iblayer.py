import numpy as np
import tensorflow as tf
import entropy

class NoisyIBLayer(tf.keras.layers.Layer):
    def __init__(self, n_noisevar_batch=1000, init_noisevar=0.01, **kwargs): # !!!! init_kdewidth=-5., **kwargs):
        # n_noisevar_batch : how many samples to use to estimate width of KDE (eta parameter)
        # init_noisevar    : initial noise variance
        # init_kdewidth    : initial width of KDE estimator
        
        super(NoisyIBLayer, self).__init__(**kwargs)
        init_phi           = np.log(np.exp(init_noisevar) - 1.).astype('float32') # softplus inverse
        
        # phi is the noise variance (in softplus space) 
        self.phi      = tf.get_variable('phi', dtype=tf.float32, trainable=True, initializer=init_phi) # !!!!
        self.noisevar = tf.nn.softplus(self.phi)

        # !!!! # eta is the kernel width for fitting the GMM, in softplus space
        # !!!!self.eta      = tf.get_variable('eta', dtype=tf.float32, trainable=False, initializer=float(init_kdewidth))
        # !!!!self.etavar   = tf.nn.softplus(self.eta)
        
        self.n_noisevar_batch = n_noisevar_batch
        
    def build(self, input_shape):
        assert len(input_shape) >= 2
        self.input_dim = int(input_shape[1])
        self.H_T_given_X = entropy.gaussian_entropy(self.input_dim, self.noisevar)
        
        self.input_spec = tf.layers.InputSpec(min_ndim=2, axes={-1: self.input_dim})
        self.built = True
        
#!!!!
# def optimize_eta(self, sess, inputs, train_X, true_outputs=None, train_Y=None):
#         # This chooses the KDE width by maximizing leave-one-out likelihood
#         # This should be called during the training loop
#         # Parameters
#         # ----------
#         # inputs       : TF placeholder for network inputs
#         # train_X      : input data
#         # true_outputs : TF placeholder for true outputs
#         # train_Y      : output data
#         x_batch = train_X[:self.n_noisevar_batch]
#         self.eta_optimizer.minimize(sess, feed_dict={inputs: x_batch}) 
        
        
    def call(self, inputs):
        self.T_nonoise     = tf.identity(inputs, name='RawInput') # useful for finding later
        
        self.dist_matrix   = entropy.pairwise_distance(inputs)
        
        # !!!!# negative log-likelihood for the 'width' of the GMM
        # !!!!self.neg_llh_eta   = entropy.GMM_negative_LLH(self.dist_matrix, self.etavar, self.input_dim)   
        # !!!!self.eta_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_llh_eta, var_list=[self.eta])

        # !!!! self.H_T_lb        = entropy.GMM_entropy(self.dist_matrix, self.noisevar + self.etavar, self.input_dim, 'lower')
        self.H_T_lb        = entropy.GMM_entropy(self.dist_matrix, self.noisevar, self.input_dim, 'lower')
        self.Ixt_lb        = self.H_T_lb - self.H_T_given_X
        
        # !!!!self.H_T           = entropy.GMM_entropy(self.dist_matrix, self.noisevar + self.etavar, self.input_dim, 'upper')
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
    