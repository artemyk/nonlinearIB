# Requires: Keras-1.2.1, tensorflow-0.12.1 or theano 0.8.2

import argparse, os, cPickle, logging
import numpy as np

parser = argparse.ArgumentParser(description='Run nonlinear IB on MNIST dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--backend', default='theano', choices=['tensorflow','theano'],
                    help='Deep learning backend to use')
parser.add_argument('--mode', choices=['regular','dropout','vIB','nlIB'], default='nlIB',
    help='Regularization mode')
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
parser.add_argument('--predict_samples', type=int, default=1, help='No. of samples to measure accuracy at end of run')
parser.add_argument('--epoch_report_mi', action='store_true', default=False, help='Report MI values every epoch?')

args = parser.parse_args()

if args.no_test_phase_noise and args.predict_samples > 1:
    raise Exception('Multiple predictions samples only makes sense if test-phase noise present')
    
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

trn, tst = buildmodel.get_mnist(args.trainN, args.testN)
# ***************************

arg_dict['INPUT_DIM'] = trn.X.shape[1]
print '# ARGS:', arg_dict

model, cbs, noiselayer, micalculator = buildmodel.buildmodel(arg_dict, trn=trn)

reporter = reporting.Reporter(trn=trn, tst=tst, noiselayer=noiselayer, micalculator=micalculator,
                             on_epoch_report_mi=args.epoch_report_mi)
cbs.append(reporter)


def lrscheduler(epoch):
    lr = 0.001 * args.lr_decay**np.floor(epoch / args.lr_decaysteps)
    lr = max(lr, 1e-5)
    print 'Learning rate: %.7f' % lr
    return lr
cbs.append(keras.callbacks.LearningRateScheduler(lrscheduler))

fit_args = dict(
    x          = trn.X,
    y          = trn.Y,
    verbose    = 2,
    batch_size = args.batch_size,
    nb_epoch   = args.nb_epoch,
    validation_data  = (tst.X, tst.Y),
    callbacks  = cbs,
)

model.compile(loss='categorical_crossentropy', optimizer=args.optimizer, metrics=['accuracy'])

hist=None
try:
    r = model.fit(**fit_args)
    hist = r.history
except KeyboardInterrupt:
    print "KeyboardInterrupt called"
    

# Print and save results
probs = 0.
for _ in range(args.predict_samples):
    probs += model.predict(tst.X)
probs /= float(args.predict_samples)
preds = probs.argmax(axis=-1)
print 'Accuracy (using %d samples): %0.5f' % (args.predict_samples, np.mean(preds == tst.y))

print '# ENDARGS:', arg_dict
print '# ENDRESULTS:',
logs = reporter.get_logs(calculate_mi=True, calculate_kl=True)
for k, v in logs.iteritems():
    print "%s=%s"%(k,v),
print


sfx = '%s-%s-%s-%f' % (args.mode, args.encoder, args.decoder, args.beta)
fname = "models/fitmodel-%s.h5"%sfx
print "saving to %s"%fname
model.save_weights(fname)

savedhistfname="models/savedhist-%s.dat"%sfx
with open(savedhistfname, 'wb') as f:
    cPickle.dump({'args':arg_dict, 'history':hist,  'endlogs': logs}, f)
    print 'updated', savedhistfname



