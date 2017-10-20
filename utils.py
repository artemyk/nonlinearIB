import numpy as np
import scipy
import keras.backend as K
import keras
from collections import namedtuple

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


