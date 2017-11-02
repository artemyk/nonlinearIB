from keras.models import Model
from keras.layers import Input, Dense, Dropout, merge
import numpy as np
import keras.datasets.mnist
import keras.utils.np_utils
from collections import namedtuple

import training
import layers
import reporting
import vib

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


def buildmodel(opts, trn):
    noiselayer   = None
    micalculator = None
    # Build model
    inputs = Input(shape=(opts['INPUT_DIM'],))
    model_layers = []
    cbs = []

    HIDDEN_DIMS = map(int, opts['encoder'].split('-'))
    HIDDEN_ACTS = opts['encoder_acts'].split('-')
    if opts['decoder']:
        DECODER_DIMS = map(int, opts['decoder'].split('-'))
    else:
        DECODER_DIMS = []
    for hndx, hdim in enumerate(HIDDEN_DIMS):
        if opts['mode'] == 'dropout':
            model_layers.append( Dropout(.2 if hndx == 0 else .5) )

        layer_args = {}    
        layer_args['activation'] = HIDDEN_ACTS[hndx]
        if layer_args['activation'] == 'relu':
            layer_args['kernel_initializer'] = 'he_uniform' 
        else:
            layer_args['kernel_initializer'] = 'glorot_uniform'
        #if args.maxnorm is not None:
        #    import keras.constraints
        #    layer_args['W_constraint'] = keras.constraints.maxnorm(args.maxnorm)

        model_layers.append( Dense(hdim, **layer_args) )

    if opts['mode'] in ['nlIB', 'nlIBnokde', 'vIB']:
        test_phase_noise = not opts['no_test_phase_noise']
        if opts['mode'] == 'nlIB' or opts['mode'] == 'nlIBnokde':
            micalculator = layers.MICalculator(opts['beta'], 
                                               model_layers, 
                                               data=trn.X, 
                                               miN=opts['miN'], 
                                               init_kde_logvar=opts['init_kde_logvar'])
            if opts['mode'] != 'nlIBnokde':
                cbs.append(training.KDETrain(mi_calculator=micalculator))
            noise_trainable = not opts.get('noise_logvar_nottrainable', False)  
            noiselayer = layers.NoiseLayer(init_logvar = opts['init_noise_logvar'], 
                                          logvar_trainable=noise_trainable,
                                          test_phase_noise=test_phase_noise)
        else:
            micalculator = vib.MICalculatorVIB(opts['beta'])
            noiselayer = vib.NoiseLayerVIB(mean_dims=HIDDEN_DIMS[-1]/2, 
                                           test_phase_noise=test_phase_noise)

        micalculator.set_noiselayer(noiselayer)

        cur_layer = inputs
        for l in model_layers:
            cur_layer = l(cur_layer)
        noise_input_layer = layers.IdentityMap(activity_regularizer=micalculator)(cur_layer)
        del cur_layer
        
        cur_layer = noiselayer(noise_input_layer)

        #if not opts['noise_logvar_grad_trainable']:
        #    cbs.append(training.NoiseTrain(traindata=trn, noiselayer=noiselayer))
    else:
        cur_layer = inputs
        for l in model_layers:
            cur_layer = l(cur_layer)

    for hndx, hdim in enumerate(DECODER_DIMS):
        layer_args = {}    
        layer_args['activation'] = 'relu' # opts['DECODING_ACTS'][hndx]
        if layer_args['activation'] == 'relu':
            layer_args['kernel_initializer'] = 'he_uniform' 
        else:
            layer_args['kernel_initializer'] = 'glorot_uniform'
        #if args.maxnorm is not None:
        #    import keras.constraints
        #    layer_args['W_constraint'] = keras.constraints.maxnorm(args.maxnorm)

        cur_layer = Dense(hdim, **layer_args)(cur_layer)

    predictions = Dense(trn.nb_classes, kernel_initializer='glorot_uniform', activation='softmax')(cur_layer)
    model = Model(inputs=inputs, outputs=predictions)
    
    return model, cbs, noiselayer, micalculator
