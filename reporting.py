from __future__ import print_function

import numpy as np
import keras.backend as K
import keras
from collections import OrderedDict

import nonlinearib, utils

def get_nlIB_layer(model):
    """ Searches through the layers of a model and returns the NonlinearIB layer, 
    if it exists.
    """
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


class Reporter(keras.callbacks.Callback):
    """ Class for reporting on progress of NonlinearIB training. Prints testing/training
    MI(X;M) (compression mutual information) and MI(M;Y) (prediction mutual information).
    Also prints (logarithm) of variance of noise and KDE.
    """
    def __init__(self, trn, tst, verbose=2):
        self.trn = trn
        self.tst = tst
        self.verbose = verbose # Verbosity level
    
    def on_train_begin(self, logs={}):
        self.nlIB_layer = get_nlIB_layer(self.model)
        inputs = self.model.inputs + self.model.targets + self.model.sample_weights + [ K.learning_phase(),]
        self.mifunc = K.function(inputs, [self.nlIB_layer.mi])
        self.h_trn = utils.np_entropy(self.trn.Y.mean(axis=0))
        self.h_tst = utils.np_entropy(self.tst.Y.mean(axis=0))
        
    def on_epoch_end(self, epoch, logs={}):
        if self.verbose > 0:
            l = self.get_logs(self.verbose > 1)
            for k in sorted(l.keys()):
                v = l[k]
                logs[k] = v
                print("%s=%s "%(k,v), sep="")
            print()

    
    def get_logs(self, include_mi=True):
        logs = {}
        logs['logvar_noise'] = np.array(K.eval(self.nlIB_layer.noise_logvar))
        logs['logvar_kde']   = np.array(K.eval(self.nlIB_layer.kde_logvar))
        
        if include_mi:
            # TODO skipping every 10ths
            trn_skip = int(len(self.trn.X)/5000)
            tst_skip = int(len(self.tst.X)/5000)
            logs['MI(X;M)_trn']  = self.mifunc([self.trn.X[::trn_skip], self.trn.Y[::trn_skip], np.ones(len(self.trn.X[::trn_skip])), 0])[0]
            logs['MI(X;M)_tst']  = self.mifunc([self.tst.X[::tst_skip], self.tst.Y[::tst_skip], np.ones(len(self.tst.X[::tst_skip])), 0])[0]
            
            # Compute cross entropy of predictions
            inputs = self.model.inputs + self.model.targets + self.model.sample_weights + [ K.learning_phase(),]
            lossfunc = K.function(inputs, [self.model.total_loss])
            logs['CrossEntropy_trn'] = lossfunc([self.trn.X, self.trn.Y, np.ones(len(self.trn.X)), 0])[0]
            logs['CrossEntropy_tst'] = lossfunc([self.tst.X, self.tst.Y, np.ones(len(self.tst.X)), 0])[0]
            logs['MI(Y;M)_trn'] = self.h_trn - logs['CrossEntropy_trn']
            logs['MI(Y;M)_tst'] = self.h_tst - logs['CrossEntropy_tst']

        return logs
        

