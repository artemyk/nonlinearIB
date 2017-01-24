# Requires: Keras-1.2.1, tensorflow-0.12.1 or theano 0.8.2

import argparse, os, cPickle, logging
import numpy as np

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
parser.add_argument('--lr_decaysteps', type=int, default=10, help='Number of iterations before dropping learning rate')
parser.add_argument('--lr_decay', type=float, default=0.5, help='Learning rate decay rate (applied every lr_decaysteps)')
parser.add_argument('--no_test_phase_noise', action='store_true', default=False, help='Disable noise during testing phase')

parser.add_argument('--encoder', type=str, default='800-800', help='Encoder network architecture')
parser.add_argument('--encoder_acts', type=str, default='relu-relu', help='Encoder layer activations')
parser.add_argument('--decoder', type=str, default='', help='Decoder network architecture')
                    
args = parser.parse_args()

if args.backend == 'theano':
    import theano
    theano.config.optimizer = 'fast_compile'
    theano.config.floatX    = 'float32'
    import os ; os.environ['KERAS_BACKEND']='theano'
else:
    import os ; os.environ['KERAS_BACKEND']='tensorflow'

logging.getLogger('keras').setLevel(logging.INFO)

import reporting
import buildmodel
import keras.callbacks

arg_dict = vars(args)

VALIDATE_ON_TEST = True
arg_dict['noise_logvar_grad_trainable'] = True


# if args.mode == 'dropout':
#     arg_dict['HIDDEN_DIMS'] = [800, 800]
#     arg_dict['HIDDEN_ACTS'] = ['relu','relu']
# elif args.mode == 'vIB':
#     #HIDDEN_DIMS = [1024, 1024, 512]
#     arg_dict['HIDDEN_DIMS'] = [1024, 1024, 4]
#     arg_dict['HIDDEN_ACTS'] = ['relu','relu', 'linear']
# elif args.mode == 'nlIB':
#     arg_dict['HIDDEN_DIMS'] = [800, 800, 2]
#     arg_dict['HIDDEN_ACTS'] = ['relu','relu', 'linear']
#     arg_dict['DECODING_DIMS'] = [800,]
#     arg_dict['DECODING_ACTS'] = ['relu',]
    
# else:
#     #raise Exception('error')
#     #arg_dict['HIDDEN_DIMS'] = [800, 800, 256, 2]
#     #arg_dict['HIDDEN_ACTS'] = ['relu','relu','relu','linear']
#     #arg_dict['DECODING_DIMS'] = [800,]
#     #arg_dict['DECODING_ACTS'] = ['relu',]
    
#     arg_dict['HIDDEN_DIMS'] = [800, 800, 2]
#     #arg_dict['HIDDEN_DIMS'] = [800, 800, 256]
#     arg_dict['HIDDEN_ACTS'] = ['relu','relu','linear']

#     arg_dict['DECODING_DIMS'] = [800,]
#     arg_dict['DECODING_ACTS'] = ['relu',]
    
#     #HIDDEN_DIMS = [800, 800, 256]
#     #HIDDEN_ACTS = ['relu','relu', 'linear']
#     #HIDDEN_DIMS = [800, 800, 256]
#     #HIDDEN_ACTS = ['relu','relu', 'linear']

    
trn, tst = buildmodel.get_mnist(args.trainN, args.testN)
# ***************************

arg_dict['INPUT_DIM'] = trn.X.shape[1]
print '# ARGS:', arg_dict

model, cbs, noiselayer, micalculator = buildmodel.buildmodel(arg_dict, trn=trn)

def lrscheduler(epoch):
    lr = 0.001 * args.lr_decay**np.floor(epoch / args.lr_decaysteps)
    lr = max(lr, 1e-5)
    print 'Learning rate: %.7f' % lr
    return lr

cbs.append(keras.callbacks.LearningRateScheduler(lrscheduler))


if VALIDATE_ON_TEST:
    validation_split = None
    validation_data = (tst.X, tst.Y)
    early_stopping = None
else:
    validation_split = 0.2
    validation_data = None
    from keras.callbacks import EarlyStopping
    cbs.append( EarlyStopping(monitor='val_loss', patience=opts['patiencelevel']) )
    
fit_args = dict(
    x          = trn.X,
    y          = trn.Y,
    verbose    = 2,
    batch_size = args.batch_size,
    nb_epoch   = args.nb_epoch,
    validation_split = validation_split,
    validation_data  = validation_data,
    callbacks  = cbs,
)

model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])

hist=None
try:
    r = model.fit(**fit_args)
    hist = r.history
except KeyboardInterrupt:
    print "KeyboardInterrupt called"
    

probs = 0.
PREDICT_SAMPLES = 12
for _ in range(PREDICT_SAMPLES):
    probs += model.predict(tst.X)
probs /= float(PREDICT_SAMPLES)
preds = probs.argmax(axis=-1)
print 'Accuracy:', np.mean(preds == tst.y)
    
# Print and save results
print '# ENDARGS:', arg_dict
print '# ENDRESULTS:',
logs = reporting.get_logs(model, trn, tst, noiselayer=noiselayer, micalculator=micalculator, MIEstimateN=args.miN)

sfx = '%s-%s-%s-%f' % (args.mode, args.encoder, args.decoder, args.beta)
fname = "models/fitmodel-%s.h5"%sfx
print "saving to %s"%fname
model.save_weights(fname)

savedhistfname="models/savedhist-%s.dat"%sfx
with open(savedhistfname, 'wb') as f:
    cPickle.dump({'args':arg_dict, 'history':hist,  'endlogs': logs}, f)
    print 'updated', savedhistfname



