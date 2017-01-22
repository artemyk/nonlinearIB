import numpy as np
import keras.backend as K
from keras.callbacks import Callback


class ReportVars(Callback):
    def __init__(self, noiselayer, *kargs, **kwargs):
        super(ReportVars, self).__init__(*kargs, **kwargs)
        self.noiselayer = noiselayer
        
    def on_epoch_end(self, epoch, logs={}):
        lv1 = K.get_value(self.noiselayer.mi_calculator.kde_logvar)
        lv2 = K.get_value(self.noiselayer.logvar)
        logs['kdeLV']   = lv1
        logs['noiseLV'] = lv2
        print 'kdeLV=%.5f, noiseLV=%.5f' % (lv1, lv2) 


from trainable import MICalculator
def get_logs(model, trn, tst, noiselayer, MIEstimateN=None):
    logs = {}

    modelobj = model.model
    inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
    lossfunc = K.function(inputs, [modelobj.total_loss])
    logs['kl_trn'] = lossfunc([trn.X, trn.Y, np.ones(len(trn.X)), 0])[0]
    logs['kl_tst'] = lossfunc([tst.X, tst.Y, np.ones(len(tst.X)), 0])[0]

    if noiselayer.mi_calculator is not None:
        lv1 = K.get_value(noiselayer.mi_calculator.kde_logvar)
        logs['kdeLV']   = lv1
        print 'kdeLV=%.5f,' % lv1,
        
    if noiselayer is not None:
        lv2 = K.get_value(noiselayer.logvar)
        logs['noiseLV'] = lv2
        print 'noiseLV=%.5f' % lv2
    
    logs['mi_trn'], logs['mi_tst'] = '-', '-'

    if noiselayer is not None and noiselayer.mi_calculator is not None:
        mitrn = trn.X
        mitst = tst.X
        if MIEstimateN is not None:
            mitrn = mitrn[np.random.choice(mitrn.shape[0], MIEstimateN), :]
            mitst = mitst[np.random.choice(mitst.shape[0], MIEstimateN), :]

        kde_logvar = K.get_value(noiselayer.mi_calculator.kde_logvar)
        noise_logvar = noiselayer.logvar
        mi_obj_trn = MICalculator(noiselayer.mi_calculator.model_layers, mitrn, init_kde_logvar=kde_logvar)
        mi_obj_tst = MICalculator(noiselayer.mi_calculator.model_layers, mitst, init_kde_logvar=kde_logvar)

        if True:
            logs['mi_trn'] =  K.function([], [mi_obj_trn.get_mi(noise_logvar), 
                                              mi_obj_trn.get_h(noise_logvar), 
                                              mi_obj_trn.get_hcond(noise_logvar)])([])
            
            logs['mi_tst'] =  K.function([], [mi_obj_tst.get_mi(noise_logvar), 
                                              mi_obj_tst.get_h(noise_logvar), 
                                              mi_obj_tst.get_hcond(noise_logvar)])([])
            
    print ', mitrn=%s, mitst=%s, kltrn=%.3f, kltst=%.3f' % (logs['mi_trn'], logs['mi_tst'], logs['kl_trn'], logs['kl_tst'])
        
    return logs

