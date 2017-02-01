import keras.backend as K
import buildmodel
import numpy as np

trn, tst = buildmodel.get_mnist()

def get_noiselayer_activations(arg_dict, fsfx, inputX, batch_size=1000):
    arg_dict['no_test_phase_noise']=True
    if 'init_kde_logvar' not in arg_dict:
        arg_dict['init_kde_logvar'] = -5.
    model, _, noiselayer, _ = buildmodel.buildmodel(arg_dict, trn)

    model.load_weights("models/fitmodel-%s.h5"%fsfx)
    
    if noiselayer is None:
        print len(model.layers)
        noiselayer = model.layers[-2]
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [noiselayer.input,])
    #pltX, pltY = trn.X[0:10000], trn.y[0:10000]
    cacts = []
    for ndx in range(0, len(inputX), batch_size):
        cacts.append( get_activations([inputX[ndx:ndx+batch_size,:],0])[0] )
    means = np.vstack(cacts)
    #cvar = np.exp(K.eval(noiselayer.logvar))
    if arg_dict['mode'] == 'vIB':
        num_dims = means.shape[1]
        means,s = means[:,:num_dims/2], means[:,num_dims/2:]
        s = np.log(1+np.exp(s - 5.))
        logvars = 2*np.log(s)
    elif arg_dict['mode'] == 'regular':
        logvars = means * 0.0
    else:
        logvars = means * 0.0 + K.eval(noiselayer.logvar)
    #logvars = 0*logvars + 0.1
    return means, logvars
