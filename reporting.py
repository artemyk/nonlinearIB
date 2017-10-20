from __future__ import print_function

import numpy as np
import keras.backend as K
from keras.callbacks import Callback
from collections import OrderedDict

import nonlinearib, utils

from keras.callbacks import Callback

class Reporter(Callback):
    def __init__(self, trn, tst):
        self.trn = trn
        self.tst = tst
    
    def on_train_begin(self, logs=None):
        self.nlIB_layer = None
        for layer in self.model.layers:
            if isinstance(layer, nonlinearib.NonlinearIB):
                if self.nlIB_layer is None:
                    self.nlIB_layer = layer
                else:
                    raise Exception('Only one NonlinearIB layer supported by Reporter')
                    
        if self.nlIB_layer is None:
            raise Exception('No NonlinearIB layers found')
            
        self.to_log = OrderedDict()
        self.to_log['noiseLV'] = self.nlIB_layer.noise_logvar
        self.to_log['kdeLV']   = self.nlIB_layer.kde_logvar
        self.to_log['mi_trn']  = self.nlIB_layer.IBpenalty(utils.tensor_constant(self.trn.X))[0]
        self.to_log['mi_tst']  = self.nlIB_layer.IBpenalty(utils.tensor_constant(self.tst.X))[0]
        
        self.h_trn = utils.np_entropy(self.trn.Y.mean(axis=0))
        self.h_tst = utils.np_entropy(self.tst.Y.mean(axis=0))
        
    def on_epoch_end(self, epoch, logs={}):
        if not len(logs):
            logs = OrderedDict()
            
        for k, v in self.to_log.items():
            logs[k] = float(K.get_value(v))
        
        # Compute cross entropy of predictions
        inputs = self.model.inputs + self.model.targets + self.model.sample_weights + [ K.learning_phase(),]
        lossfunc = K.function(inputs, [self.model.total_loss])
        logs['loss_trn'] = lossfunc([self.trn.X, self.trn.Y, np.ones(len(self.trn.X)), 0])[0]
        logs['loss_tst'] = lossfunc([self.tst.X, self.tst.Y, np.ones(len(self.tst.X)), 0])[0]
        logs['loss_mi_trn'] = self.h_trn - logs['loss_trn']
        logs['loss_mi_tst'] = self.h_tst - logs['loss_tst']
        
        for k, v in logs.items():
            logs[k]=v
            print("%s=%s "%(k,v), sep="")
        print()
        
