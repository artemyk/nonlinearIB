from __future__ import print_function

import numpy as np
import keras.backend as K
from keras.callbacks import Callback
import copy
from collections import OrderedDict
from entropy import np_entropy

#nats2bits = np.array(1.0/np.log(2), dtype='float32')


class Reporter(Callback):
    def __init__(self, trn, tst, noiselayer, micalculator, on_epoch_report_mi=False):
        self.trn = trn
        self.tst = tst
        self.noiselayer = noiselayer
        self.micalculator = micalculator
        self.on_epoch_report_mi = on_epoch_report_mi
        
    def on_epoch_end(self, epoch, logs={}):
        l = self.get_logs(calculate_mi=self.on_epoch_report_mi)
        for k, v in l.items():
            logs[k]=v
            print("%s=%s "%(k,v), sep="")
        print()
    
    def get_logs(self, calculate_mi=False, calculate_loss=False):
        logs = OrderedDict()
        
        if self.noiselayer is not None and hasattr(self.noiselayer, 'logvar'):
            logs['noiseLV'] = K.get_value(self.noiselayer.logvar)

        inputs = self.model.inputs + self.model.targets + self.model.sample_weights + [ K.learning_phase(),]
        trn_inputs = [self.trn.X, self.trn.Y, np.ones(len(self.trn.X)), 0]
        tst_inputs = [self.tst.X, self.tst.Y, np.ones(len(self.tst.X)), 0]
        
        if self.micalculator is not None and hasattr(self.micalculator, 'kde_logvar'):
                logs['kdeLV'] = K.get_value(self.micalculator.kde_logvar)
                
        if self.micalculator is not None and calculate_mi:
            f = K.function(inputs, [self.noiselayer.input])
            noiselayer_inputs = {}
            noiselayer_inputs['trn']  = f(trn_inputs)[0]
            noiselayer_inputs['tst']  = f(tst_inputs)[0]

            for k in ['trn','tst']:
                mi_calc = self.micalculator
                if k != 'trn' and hasattr(mi_calc, 'set_data'):
                    mi_calc = copy.copy(mi_calc)
                    mi_calc.set_data(self.tst.X)

                h, hcond = 0., 0.
                c_in = K.variable(noiselayer_inputs[k])
                mi = K.function([], [mi_calc.get_mi(c_in)])([])[0]
                #if hasattr(mi_calc, 'get_h'):
                #    h     = K.function([], [mi_calc.get_h(c_in)])([])[0]
                #if hasattr(mi_calc, 'get_hcond'):
                #    hcond = K.function([], [mi_calc.get_hcond(c_in)])([])[0]
                #logs['mi_'+k] = map(float, [mi, h, hcond])
                logs['mi_'+k] = float(mi)

        if calculate_loss:
            h_trn = np_entropy(self.trn.Y.mean(axis=0))
            h_tst = np_entropy(self.tst.Y.mean(axis=0))
            # Compute cross entropy of predictions
            lossfunc = K.function(inputs, [self.model.total_loss])
            logs['loss_trn'] = lossfunc(trn_inputs)[0]
            logs['loss_tst'] = lossfunc(tst_inputs)[0]
            logs['loss_mi_trn'] = h_trn - logs['loss_trn']
            logs['loss_mi_tst'] = h_tst - logs['loss_tst']
        
        return logs
