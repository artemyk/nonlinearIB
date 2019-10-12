import signal
import logging
import os
import pickle
import numpy as np


class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)

def write_data(savefile, saveobjs):
    with DelayedKeyboardInterrupt():
        savedir = os.path.dirname(savefile)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        with open(savefile, 'wb') as fp:
            pickle.dump(saveobjs, fp)
            
            
            
def softplusinverse(x):
    return np.log(1-np.exp(-x)) + x # numerically stable inverse of softplus

                  
    
def get_train_batches(X, Y, batchsize):
    N = len(Y)
    n_mini_batches = int(np.ceil(N / batchsize))

    # randomize order of training data
    permutation  = np.random.permutation(N)
    rndX = X[permutation]
    rndY = Y[permutation]

    batches = [ {'X:0'     : rndX[batch * batchsize:(1 + batch) * batchsize], 
                 'trueY:0' : rndY[batch * batchsize:(1 + batch) * batchsize]}
                for batch in range(n_mini_batches) ]
    return batches


def get_error(errtype, y_true, y_pred): # cross entropy or mse
    import tensorflow as tf
    if errtype == 'ce':
        f = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
    elif errtype == 'mse':
        f = (y_true - y_pred)**2
    else:
        raise Exception('Unknown errtype', errtype)
    return tf.reduce_mean(f)

def get_accuracy(errtype, y_true, y_pred): # cross entropy or mse
    import tensorflow as tf
    if errtype == 'ce':
        return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1)), tf.float32))
    elif errtype == 'mse':
        return tf.constant(np.nan)
    else:
        raise Exception('Unknown errtype', errtype)

        
