import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import entropy
ds = tf.contrib.distributions


class Net(object):
    
    def get_noisevar_optimizer(self, loss):
        return tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[self.phi])
        # return tfp.optimizer.bfgs_minimize(loss, var_list=[self.phi,])
    
    def get_kdewidth_optimizer(self, loss):
        return tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[self.eta])
        # return tfp.optimizer.bfgs_minimize(loss, var_list=[self.eta,])
    
    def __init__(self, input_dims, encoder_arch, decoder_arch, err_func, entropyY, trainable_noisevar = True, 
                 noisevar = 0., kdewidth=-5):
        hiddenD       = encoder_arch[-1][0]  # bottleneck layer dimensionality

        self.x        = tf.placeholder(dtype=tf.float32, name='X', shape=[None, input_dims])           # inputs
        self.y        = tf.placeholder(dtype=tf.float32, name='Y', shape=[None, decoder_arch[-1][0]])  # outputs
        
        # phi is the noise variance (in softplus space) 
        init_phi      = np.log(np.exp(noisevar) - 1.).astype('float32') # softplus inverse
        self.trainable_noisevar = trainable_noisevar
        if trainable_noisevar:
            self.phi  = tf.get_variable('phi', dtype=tf.float32, trainable=True, initializer=init_phi)
        else:
            self.phi  = tf.get_variable('phi', dtype=tf.float32, trainable=False, initializer=init_phi)
        self.noisevar = tf.nn.softplus(self.phi)
        
        # Only used if fitting noise variance with outer optimization loop, not SGD batches
        
        
        # START building the neural network
        self.encoder  = [self.x,]
        for endx, (neurons, act) in enumerate(encoder_arch):
            self.encoder.append(tf.layers.dense(self.encoder[-1], neurons, name='encoder_%d'%endx, activation=act))
            
        tf.identity(self.encoder[-1], name="T_nonoise")
            
        noise         = tf.random_normal(shape=tf.shape(self.encoder[-1]), 
                                         mean=0.0, stddev=tf.sqrt(self.noisevar), dtype=tf.float32)
        self.T        = self.encoder[-1] + noise # hidden layer with noise
        
        self.decoder  = [self.T,]
        for dndx, (neurons, act) in enumerate(decoder_arch):
            self.decoder.append(tf.layers.dense(self.decoder[-1], neurons, name='decoder_%d'%dndx, activation=act))
            
        self.predY    = self.decoder[-1]
        # END building the neural network
        
        self.distance_matrix = entropy.pairwise_distance(self.encoder[-1])
        
        
        # eta is the kernel width for fitting the GMM, in softplus space
        self.eta      = tf.get_variable('eta', dtype=tf.float32, trainable=False, initializer=kdewidth)
        self.etavar   = tf.nn.softplus(self.eta)
        # # placeholder to speed up scipy optimizer
        # self.distance_matrix_ph = tf.placeholder(dtype=tf.float32, shape=[None, None]) 
        # negative log-likelihood for the 'width' of the GMM
        self.neg_llh_eta        = entropy.GMM_negative_LLH(self.distance_matrix, self.etavar, hiddenD)   
        self.eta_optimizer      = self.get_kdewidth_optimizer(self.neg_llh_eta)
        
        
        if err_func == 'softmax_ce':
            f = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.predY)
        elif err_func == 'mse':
            f = tf.losses.mean_squared_error(self.y, self.predY)
        else:
            raise Exception('unknown err_func')
            
        self.cross_entropy   = tf.reduce_mean(f)
        self.accuracy        = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predY, 1), tf.argmax(self.y, 1)), tf.float32))
        self.Iyt             = entropyY - self.cross_entropy

        self.H_T_given_X = entropy.gaussian_entropy(hiddenD, self.noisevar)
        self.H_T_lb      = entropy.GMM_entropy(self.distance_matrix, self.noisevar + self.etavar, hiddenD, 'lower')
        self.Ixt_lb      = self.H_T_lb - self.H_T_given_X
        
        self.H_T         = entropy.GMM_entropy(self.distance_matrix, self.noisevar + self.etavar, hiddenD, 'upper')
        self.Ixt         = self.H_T - self.H_T_given_X

        # Variational IB code, based on Alemi
        # https://github.com/alexalemi/vib_demo/blob/master/MNISTVIB.ipynb
        prior            = ds.Normal(0.0, 1.0)
        encoding         = ds.Normal(self.encoder[-1], self.noisevar)
        self.vIxt        = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(encoding, prior), 0))
        
        self.vH_T        = self.vIxt + self.H_T_given_X
        
        
        # list of network parameters, excluding ones changing variance / kernel width
        self.no_var_params = [v for v in tf.trainable_variables() if v not in [self.phi, self.eta]]
        
