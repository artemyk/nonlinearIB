import numpy as np

def randsample(mx, maxN, axis=0, replace=False, return_ixs=False):
    if axis not in [0,1]:
        raise Exception('axis should be in [0,1]')
    ixs = np.random.choice(mx.shape[axis], size=maxN, replace=replace)
    r = mx[ixs,:] if axis == 0 else mx[:,ixs]
    if return_ixs:
        return r, ixs
    else:
        return r

def get_logs(model, data, kdelayer, noiselayer, max_entropy_calc_N=None):
    logs = {}

    modelobj = model.model
    inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
    lossfunc = K.function(inputs, [modelobj.total_loss])
    sampleweightstrn = np.ones(len(data.train.X))
    sampleweightstst = np.ones(len(data.test.X))
    noreglosstrn = lambda: lossfunc([data.train.X, data.train.Y, sampleweightstrn, 0])[0]
    noreglosstst = lambda: lossfunc([data.test.X , data.test.Y , sampleweightstst, 0])[0]

    if kdelayer is not None:
        lv1 = K.get_value(kdelayer.logvar)
        logs['kdeLV']   = lv1
        print 'kdeLV=%.5f,' % lv1,
        
    if noiselayer is not None:
        lv2 = K.get_value(noiselayer.logvar)
        logs['noiseLV'] = lv2
        print 'noiseLV=%.5f' % lv2
    
    if kdelayer is not None and noiselayer is not None:
        if max_entropy_calc_N is None:
            mitrn = data.train.X
            mitst = data.test.X
        else:
            mitrn = randsample(data.train.X, max_entropy_calc_N)
            mitst = randsample(data.test.X, max_entropy_calc_N)

        mi_obj_trn = MIComputer(noiselayer.get_noise_input_func(mitrn), kdelayer=kdelayer, noiselayer=noiselayer)
        mi_obj_tst = MIComputer(noiselayer.get_noise_input_func(mitst), kdelayer=kdelayer, noiselayer=noiselayer)

        if True:
            def eval_with_training(x):
                return float(K.function([K.learning_phase()], [x])([True,])[0])
            
            mivals_trn = map(eval_with_training, [mi_obj_trn.get_mi(), mi_obj_trn.get_h(), mi_obj_trn.get_hcond()])
            logs['mi_trn'] = mivals_trn[0]
            
            # Even with testing data, evaluate with training_phase so that noise is injected (gets better esitmate of MI)
            mivals_tst = map(eval_with_training, [mi_obj_tst.get_mi(), mi_obj_tst.get_h(), mi_obj_tst.get_hcond()])
            
            logs['mi_tst'] = mivals_tst[0]
            logs['kl_trn'] = noreglosstrn()
            logs['kl_tst'] = noreglosstst()
            print ', mitrn=%s, mitst=%s, kltrn=%.3f, kltst=%.3f' % (mivals_trn, mivals_tst, logs['kl_trn'], logs['kl_tst'])
        else:
            print
        
    return logs

