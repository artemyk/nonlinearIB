import tensorflow as tf
import numpy as np
import entropy
ds = tf.contrib.distributions

class Loss(object):
    def __init__(self, f, var_list=None):
        self.f = f
        self.var_list = var_list
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) # 
        self.trainstep = self.optimizer.minimize(self.f, var_list=var_list)
        
class Net(object):
    def __init__(self, input_dims, encoder_arch, decoder_arch, err_func, entropyY, 
                 init_beta = 0.0, squaredIB=False,
                 gradient_train_noisevar = True, noisevar = 0.):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, input_dims]) # digit images
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, decoder_arch[-1][0]])  # one-hot labels
        hiddenD       = encoder_arch[-1][0]
        
        self.beta       = tf.get_variable('beta', dtype=tf.float32, trainable=False, initializer=float(init_beta))
        
        self.gradient_train_noisevar = gradient_train_noisevar
        init_phi = np.log(np.exp(noisevar) - 1.).astype('float32') # softplus inverse
        if gradient_train_noisevar:
            self.phi    = tf.get_variable('phi', dtype=tf.float32, trainable=True, initializer=init_phi)
        else:
            self.phi    = tf.get_variable('phi', dtype=tf.float32, trainable=False, initializer=init_phi)
        self.noisevar = tf.nn.softplus(self.phi)
        
        if True:
            self.eta      = tf.get_variable('eta', dtype=tf.float32, trainable=False, initializer=-5.)
            self.etavar   = tf.nn.softplus(self.eta)
            # for fitting the GMM
            self.distance_matrix_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])  
            # placeholder to speed up scipy optimizer
            self.neg_llh_eta = entropy.GMM_negative_LLH(self.distance_matrix_ph, self.etavar, hiddenD)   
            # negative log-likelihood for the 'width' of the GMM
            self.eta_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_llh_eta, var_list=[self.eta])

            
        self.encoder = [self.x,]
        for neurons, activation in encoder_arch:
            self.encoder.append(tf.layers.dense(self.encoder[-1], neurons, activation=activation))
        noise         = tf.random_normal(shape=tf.shape(self.encoder[-1]), 
                                         mean=0.0, stddev=tf.sqrt(self.noisevar), dtype=tf.float32)
        self.T        = self.encoder[-1] + noise
        self.decoder  = [self.T,]
        for neurons, activation in decoder_arch:
            self.decoder.append(tf.layers.dense(self.decoder[-1], neurons, activation=activation))
            
        self.predY    = self.decoder[-1]
        
        self.distance_matrix = entropy.pairwise_distance(self.encoder[-1])
        
        if err_func == 'softmax_ce_v2':
            f = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y, logits=self.predY)
        elif err_func == 'mse':
            f = tf.losses.mean_squared_error(self.y, self.predY)
        else:
            raise Exception('unknown err_func')
        self.cross_entropy   = tf.reduce_mean(f)
        self.accuracy        = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predY, 1), tf.argmax(self.y, 1)), tf.float32))

        self.H_T = entropy.GMM_entropy(self.distance_matrix, self.noisevar + self.etavar, hiddenD, 'upper')
        self.H_T_given_X = entropy.Gaussian_entropy(hiddenD, self.noisevar)
        self.Ixt = self.H_T - self.H_T_given_X
        
        self.H_T_lb = entropy.GMM_entropy(self.distance_matrix, self.noisevar + self.etavar, hiddenD, 'lower')
        self.Ixt_lb = self.H_T_lb - self.H_T_given_X

        
        self.Iyt = entropyY - self.cross_entropy
        self.nlIB_loss           = Loss(self.beta*(self.Ixt**2 if squaredIB else self.Ixt) - self.Iyt)

        prior = ds.Normal(0.0, 1.0)
        encoding = ds.Normal(self.encoder[-1], self.noisevar)
        self.vIxt = tf.reduce_sum(tf.reduce_mean(ds.kl_divergence(encoding, prior), 0))
        self.VIB_loss           = Loss(self.beta*(self.vIxt**2 if squaredIB else self.vIxt) - self.Iyt)

        vl = [v for v in tf.trainable_variables() if v != self.phi]
        self.ce_loss            = Loss(self.cross_entropy, var_list=vl)
        
