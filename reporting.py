import numpy as np
import keras.backend as K
from keras.callbacks import Callback
import copy

class ReportVars(Callback):
    def __init__(self, noiselayer, micalculator, *kargs, **kwargs):
        super(ReportVars, self).__init__(*kargs, **kwargs)
        self.noiselayer = noiselayer
        self.micalculator = micalculator
        
    def on_epoch_end(self, epoch, logs={}):
        lv1, lv2 = 0., 0.
        if hasattr(self.micalculator, 'kde_logvar'):
            lv1 = K.get_value(self.micalculator.kde_logvar)
        if hasattr(self.noiselayer, 'logvar'):
            lv2 = K.get_value(self.noiselayer.logvar)
        logs['kdeLV']   = lv1
        logs['noiseLV'] = lv2
        print 'kdeLV=%.5f, noiseLV=%.5f' % (lv1, lv2) 


from layers import MICalculator
def get_logs(model, trn, tst, noiselayer, micalculator, MIEstimateN=None):
    logs = {}

    inputs = model.inputs + model.targets + model.sample_weights + [ K.learning_phase(),]
    lossfunc = K.function(inputs, [model.total_loss])
    logs['kl_trn'] = lossfunc([trn.X, trn.Y, np.ones(len(trn.X)), 0])[0]
    logs['kl_tst'] = lossfunc([tst.X, tst.Y, np.ones(len(tst.X)), 0])[0]

    if micalculator is not None:
        if hasattr(micalculator, 'kde_logvar'):
            lv1 = K.get_value(micalculator.kde_logvar)
            logs['kdeLV']   = lv1
            print 'kdeLV=%.5f,' % lv1,
        
    if noiselayer is not None and hasattr(noiselayer, 'logvar'):
        lv2 = K.get_value(noiselayer.logvar)
        logs['noiseLV'] = lv2
        print 'noiseLV=%.5f' % lv2
    
    logs['mi_trn'], logs['mi_tst'] = '-', '-'

    if micalculator is not None:
        lossfunc = K.function(inputs, [noiselayer.input])
        noiselayer_inputs = {}
        noiselayer_inputs['trn']  = lossfunc([trn.X, trn.Y, np.ones(len(trn.X)), 0])[0]
        noiselayer_inputs['tst']  = lossfunc([tst.X, tst.Y, np.ones(len(tst.X)), 0])[0]
        
        for k in ['trn','tst']:
            mi_calc = micalculator
            if k != 'trn' and hasattr(mi_calc, 'set_data'):
                mi_calc = copy.copy(mi_calc)
                mi_calc.set_data(tst.X)
                                
            h, hcond = 0., 0.
            c_in = K.variable(noiselayer_inputs[k])
            mi = K.function([], [mi_calc.get_mi(c_in)])([])[0]
            if hasattr(mi_calc, 'get_h'):
                h     = K.function([], [mi_calc.get_h(c_in)])([])[0]
            if hasattr(mi_calc, 'get_hcond'):
                hcond = K.function([], [mi_calc.get_hcond(c_in)])([])[0]
                
            logs['mi_'+k] = map(float, [mi, h, hcond])
            
    print ', mitrn=%s, mitst=%s, kltrn=%.3f, kltst=%.3f' % (logs['mi_trn'], logs['mi_tst'], logs['kl_trn'], logs['kl_tst'])
        
    return logs

