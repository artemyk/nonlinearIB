# Requires: Keras-1.2.1, tensorflow-0.12.1 or theano 0.8.2

import argparse

parser = argparse.ArgumentParser(description='Run nonlinear IB on MNIST dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--backend', default='theano', choices=['tensorflow','theano'],
                    help='Deep learning backend to use')
parser.add_argument('--mode', choices=['regular','dropout','vIB','nlIB'], default='nlIB',
    help='Regularization mode')
parser.add_argument('--nb_mc_samples', type=int, default=1, help='Number of MC samples')
parser.add_argument('--nb_epoch', type=int, default=60, help='Number of epochs')
parser.add_argument('--beta' , type=float, default=0.0, help='beta hyperparameter value')
parser.add_argument('--init_noise_logvar', type=float, default=-6., help='Initialize log variance of noise')
#parser.add_argument('--maxnorm', type=float, help='Max-norm constraint to impose')
parser.add_argument('--trainN', type=int, help='Number of training data samples')
parser.add_argument('--testN', type=int, help='Number of testing data samples')
parser.add_argument('--miN', type=int, default=1000, help='Number of training data samples to use for estimating MI')
parser.add_argument('--batch_size', type=int, default=128, help='Mini-batch size')
parser.add_argument('--optimizer', choices=['sgd','rmsprop','adagrad','adam'], default='adam',
                    help='Which optimizer to use')
parser.add_argument('--lr_halftime', type=int, default=10, help='Number of iterations before dropping learning rate in half')

args = parser.parse_args()

print '# ARGS:', vars(args)

import os
if args.backend == 'theano':
    import theano
    theano.config.optimizer = 'fast_compile'
    theano.config.floatX    = 'float32'
    import os ; os.environ['KERAS_BACKEND']='theano'
else:
    import os ; os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np

from collections import namedtuple
import keras
import keras.datasets.mnist
import keras.utils.np_utils
from keras.models import Model
from keras.layers import Input, Dense, Dropout, merge
import logging    
logging.getLogger('keras').setLevel(logging.INFO)

import training
import layers
import reporting

VALIDATE_ON_TEST = True
noise_logvar_grad_trainable = True

if args.mode == 'dropout':
    HIDDEN_DIMS = [800, 800]
    HIDDEN_ACTS = ['relu','relu']
elif args.mode == 'vIB':
    #HIDDEN_DIMS = [1024, 1024, 512]
    HIDDEN_DIMS = [1024, 1024, 4]
    
    HIDDEN_ACTS = ['relu','relu', 'linear']
else:
    HIDDEN_DIMS = [800, 800, 2]
    #HIDDEN_DIMS = [800, 800, 256]
    #HIDDEN_ACTS = ['relu','relu', 'linear']
    HIDDEN_ACTS = ['tanh','tanh','linear']

print 'HIDDEN_DIMS=%s, HIDDEN_ACTS=%s' % (HIDDEN_DIMS, HIDDEN_ACTS)

# Initialize MNIST dataset
nb_classes = 10
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = np.reshape(X_train, [X_train.shape[0], -1]).astype('float32') / 255.
X_test  = np.reshape(X_test , [X_test.shape[0] , -1]).astype('float32') / 255.
Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes)
Y_test  = keras.utils.np_utils.to_categorical(y_test, nb_classes)


if args.trainN is not None:
    X_train = X_train[0:args.trainN]
    Y_train = Y_train[0:args.trainN]

if args.testN is not None:
    X_test = X_test[0:args.testN]
    Y_test = Y_test[0:args.testN]


Dataset = namedtuple('Dataset',['X','Y','nb_classes'])
trn = Dataset(X_train, Y_train, nb_classes)
tst = Dataset(X_test , Y_test, nb_classes)

DIMS = trn.X.shape[1]

del X_train, X_test, Y_train, Y_test, y_train, y_test
# ***************************


# Build model
inputs = Input(shape=(DIMS,))
model_layers = []

for hndx, hdim in enumerate(HIDDEN_DIMS):
    if args.mode == 'dropout':
        model_layers.append( Dropout(.2 if hndx == 0 else .5) )
            
    layer_args = {}    
    layer_args['activation'] = HIDDEN_ACTS[hndx]
    if layer_args['activation'] == 'relu':
        layer_args['init'] = 'he_uniform' 
    else:
        layer_args['init'] = 'glorot_uniform'
    #if args.maxnorm is not None:
    #    import keras.constraints
    #    layer_args['W_constraint'] = keras.constraints.maxnorm(args.maxnorm)

    model_layers.append( Dense(hdim, **layer_args) )

def lrscheduler(epoch):
    lr = 0.001 * 0.5**np.floor(epoch / args.lr_halftime)
    lr = max(lr, 1e-5)
    print 'Learning rate: %.7f' % lr
    return lr

cbs = [keras.callbacks.LearningRateScheduler(lrscheduler),]

if args.mode in ['nlIB', 'vIB']:
    
    mi_samples = trn.X       # input samples to use for estimating 
                             # mutual information b/w input and hidden layers
    rows = np.random.choice(mi_samples.shape[0], args.miN)
    mi_samples = mi_samples[rows,:]

    if args.mode == 'nlIB':
        micalculator = layers.MICalculator(args.beta, model_layers, input_samples=mi_samples, init_kde_logvar=-5.)
        cbs.append(training.KDETrain(mi_calculator=micalculator))
        noiselayer = layers.NoiseLayer(init_logvar = args.init_noise_logvar, 
                                    logvar_trainable=noise_logvar_grad_trainable)
        #,
        #                            activity_regularizer=micalculator)
    else:
        micalculator = layers.MICalculatorVIB(args.beta)
        noiselayer = layers.NoiseLayerVIB(mean_dims=HIDDEN_DIMS[-1]/2, test_phase_noise=True)
        
    micalculator.set_noiselayer(noiselayer)
    
    curlayer = inputs
    for l in model_layers:
        curlayer = l(curlayer)
    noise_input_layer = layers.IdentityMap(activity_regularizer=micalculator)(curlayer)
    del curlayer
    
    #decoding_hidden_layer = None
    #decoding_hidden_layer = Dense(trn.nb_classes, init='he_uniform', activation='relu')
    prediction_layer = Dense(trn.nb_classes, init='glorot_uniform', activation='softmax')
    if args.nb_mc_samples > 1:
        predictions = []
        targets = []
        targets_val = []
        for ndx in range(args.nb_mc_samples):
            predictions.append(prediction_layer(noiselayer(noise_input_layer)))
            targets.append(trn.Y)
            targets_val.append(tst.Y)
    else:
        #cur_layer = 
        predictions = prediction_layer(noiselayer(noise_input_layer))
            
        targets = trn.Y
        targets_val = tst.Y
        
    #if not opts['noise_logvar_grad_trainable']:
    #    cbs.append(training.NoiseTrain(traindata=trn, noiselayer=noiselayer))
    cbs.append(reporting.ReportVars(noiselayer=noiselayer, micalculator=micalculator))

else:
    curlayer = inputs
    for l in model_layers:
        curlayer = l(curlayer)
    predictions = Dense(trn.nb_classes, init='glorot_uniform', activation='softmax')(curlayer)


if VALIDATE_ON_TEST:
    validation_split = None
    validation_data = (tst.X, targets_val)
    early_stopping = None
else:
    validation_split = 0.2
    validation_data = None
    from keras.callbacks import EarlyStopping
    cbs.append( EarlyStopping(monitor='val_loss', patience=opts['patiencelevel']) )
    
fit_args = dict(
    x          = trn.X,
    y          = targets,
    verbose    = 2,
    batch_size = args.batch_size,
    nb_epoch   = args.nb_epoch,
    validation_split = validation_split,
    validation_data  = validation_data,
    callbacks  = cbs,
)

model = Model(input=inputs, output=predictions)
model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])
    
hist = model.fit(**fit_args)

# Print and save results
print '# ENDARGS:', vars(args)
print '# ENDRESULTS:',
logs = reporting.get_logs(model, trn, tst, noiselayer=noiselayer, micalculator=micalculator, MIEstimateN=args.miN)

fname = "models/fitmodel-%s-%0.5f.h5"%(args.mode,args.beta)
print "saving to %s"%fname
model.save_weights(fname)

savedhistfname='models/savedhist-%s-%0.5f.h5"%(args.mode,args.beta)
with open(savedhistfname, 'wb') as f:
    cPickle.dump({'history':hist.history,  'endlogs': logs}, f)
    print 'updated', savedhistfname



