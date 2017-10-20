import numpy as np
import scipy
import scipy.optimize

import keras.backend as K
import keras
from collections import namedtuple


# TODO: Describe
class ParameterTrainer(keras.callbacks.Callback):
    def __init__(self, loss, parameter, trn, minibatchsize, *kargs, **kwargs):
        super(ParameterTrainer, self).__init__(*kargs, **kwargs)
        self.loss = loss
        self.parameter = parameter
        self.trn = trn
        self.minibatchsize = minibatchsize
        

    def on_train_begin(self, logs={}):
        # Loss and parameter should be defined by this point
        inputs = self.model.inputs + self.model.targets + self.model.sample_weights + [ K.learning_phase(),]
        f_obj = K.function(inputs, [self.loss,])
        f_jac = K.function(inputs, [K.gradients(self.loss, self.parameter),])
        
        def getixs(x):
            k = x.flat[0]
            if k not in self.random_samples:
                self.random_samples[k] = np.random.choice(len(self.trn.X), self.minibatchsize)
            return self.random_samples[k]
        
        def obj(x):
            ixs = getixs(x)
            oldval = K.get_value(self.parameter)
            K.set_value(self.parameter, x.flat[0])
            r = f_obj([self.trn.X[ixs], self.trn.Y[ixs], np.ones(self.minibatchsize), 1])[0]
            K.set_value(self.parameter, oldval)
            return r
        
        def jac(x):
            ixs = getixs(x)
            oldval = K.get_value(self.parameter)
            K.set_value(self.parameter, x.flat[0])
            r = f_jac([self.trn.X[ixs], self.trn.Y[ixs], np.ones(self.minibatchsize), 1])
            K.set_value(self.parameter, oldval)
            return np.atleast_2d(np.array(r[0]))[0]
        
        self.obj = obj
        self.jac = jac
        
    def on_epoch_begin(self, epoch, logs={}):
        self.random_samples = {}
        r = scipy.optimize.minimize(self.obj, K.get_value(self.parameter).flat[0], jac=self.jac)
        best_val = r.x.flat[0]
        K.set_value(self.parameter, best_val)
        del self.random_samples

        

def np_entropy(p):
    cp = np.log(p)
    cp[np.isclose(p,0.)]=0.
    return -p.dot(cp)


def logsumexp(mx, axis):
    cmax = K.max(mx, axis=axis)
    cmax2 = K.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + K.log(K.sum(K.exp(mx2), axis=1))


def dist_mx(X):
    x2 = K.expand_dims(K.sum(K.square(X), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(X, K.transpose(X))
    return dists


def get_input_layer(clayer):
    if not hasattr(clayer, 'inbound_nodes') or len(clayer.inbound_nodes) == 0 or \
       len(clayer.inbound_nodes[0].inbound_layers) == 0:
        return None
    if len(clayer.inbound_nodes) > 1:
        raise Exception("Currently doesn't work with multi-input layers")
    input_node = clayer.inbound_nodes[0]
    if len(input_node.inbound_layers) > 1:
        raise Exception("Currently doesn't work with multi-input layers")
    return input_node.inbound_layers[0]


def get_mnist(trainN=None, testN=None):
    # Initialize MNIST dataset
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.
    X_test  = np.reshape(X_test , [X_test.shape[0] , -1]).astype('float32') / 255.
    X_train = X_train * 2.0 - 1.0
    X_test  = X_test  * 2.0 - 1.0

    if trainN is not None:
        X_train = X_train[0:trainN]
        y_train = y_train[0:trainN]

    if testN is not None:
        X_test = X_test[0:testN]
        y_test = y_test[0:testN]

    Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes)
    Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes)

    Dataset = namedtuple('Dataset',['X','Y','y','nb_classes'])
    trn = Dataset(X_train, Y_train, y_train, nb_classes)
    tst = Dataset(X_test , Y_test, y_test, nb_classes)

    del X_train, X_test, Y_train, Y_test, y_train, y_test
    
    return trn, tst


def get_nlIB_layer(model):
    from nonlinearib import NonlinearIB
    nlIB_layer = None
    for layer in model.layers:
        if isinstance(layer, NonlinearIB):
            if nlIB_layer is None:
                nlIB_layer = layer
            else:
                raise Exception('Only one NonlinearIB layer supported by Reporter')

    if nlIB_layer is None:
        raise Exception('No NonlinearIB layers found')
        
    return nlIB_layer




# Backend specific code

if K._BACKEND == 'tensorflow':
    import tensorflow as tf
    def K_n_choose_k(n, k, seed=None):
        if seed is None:
            seed = np.random.randint(10e6)
        x = tf.range(0, limit=n, dtype='int32')
        x = tf.random_shuffle(x, seed=seed)
        x = x[0:k]
        return x

    def tensor_eye(size):
        return tf.eye(size)

    def tensor_constant(x):
        return tf.constant(x) 

    
else:
    import theano.tensor as T
    def K_n_choose_k(n, k, seed=None):
        from theano.tensor.shared_randomstreams import RandomStreams
        if seed is None:
            seed = np.random.randint(1, 10e6)
        rng = RandomStreams(seed=seed)
        r = rng.choice(size=(k,), a=n, replace=False, dtype='int32')
        return r

    def tensor_eye(size):
        return T.eye(size)

    def tensor_constant(x):
        return K.variable(x)


