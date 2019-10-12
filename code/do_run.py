# Script to generate nonlinearIB and VIB models for MNIST, FashionMNIST, and Housing datasets


import numpy as np
import matplotlib.pyplot as plt
import pathlib, os, time, argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)

parser.add_argument("-objective", default='sq', choices=['reg','sq'],
                                  help="Use regular or squared (I(Y:T) - beta*I(X:T)^2) IB objective")
parser.add_argument("-batchsize", help="Batch size", default=128, type=int)
parser.add_argument("-initnoise", help="Init noisevar", default=1, type=float)
parser.add_argument("-n_runs"   , help="How many runs to do", default=1, type=int)
parser.add_argument("-n_hidden" , help="How many neurons in bottleneck layer", default=10, type=int)
parser.add_argument("-n_wide"   , help="How many neurons in encoding/decoding layers", default=128, type=int)
parser.add_argument("-beta_min" , help="Min beta value", default=0.001, type=float)
parser.add_argument("-beta_max" , help="Max beta value", default=2, type=float)
parser.add_argument("-beta_npoints", help="Number beta values to sweep", default=20, type=int)
parser.add_argument("-patience"  , default=10, type=int,
                                  help="How many epochs before validation improvement for early stoppoing")
parser.add_argument('-methods',   help="Only run nlIB/VIB/ce (baseline)", default='ce,nlIB,VIB', type=str)
parser.add_argument('-report_every', default=10, type=int, help='How often to print stats')
parser.add_argument('runtype',    help="Which dataset", choices=['MNIST','FashionMNIST','Housing','Wine'])
parser.add_argument('outputdir',  help="Where to store output files", type=str)
args = parser.parse_args()

cfg          = vars(args)  # Save configuration options
    
savedirbase  = str(pathlib.Path().absolute()) + '/' + args.outputdir + '/'
savedir      = savedirbase + cfg['runtype'] 
betavals     = np.geomspace(cfg['beta_min'], cfg['beta_max'], cfg['beta_npoints'], endpoint=True)[::-1]
run_methods  = cfg['methods'].split(',')    # only run specified method 


if not os.path.exists(savedir):
    print("Making directory", savedir)
    os.makedirs(savedir)

    
import loaddata, iblayer, utils
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth=True

data         = loaddata.load_data(cfg['runtype'], validation=True)
input_dim    = data['trn_X'].shape[1]
output_dim   = data['trn_Y'].shape[1]


def train(sess, method, beta, cfg, data, net, savedir):
    # sess         : TensorFlow session
    # method       : 'ce' (cross-entropy only), 'nlIB' (nonlinear IB), or 'VIB' (variational IB)
    # beta         : beta value
    # cfg          : configuration dictionary
    # data         : data object
    # net          : neural network object
    # savedir      : directory where to save results
    
    def calcstats(epoch, do_print=False):
        lobj     = net.iblayerobj
        noisevar = sess.run(lobj.noisevar)
        cdata    = {'method': method, 'epoch': epoch, 'beta': beta, 'noisevar': noisevar} 

        for trntstmode in ['trn', 'val', 'tst']:
            X, Y = data[trntstmode+'_X'], data[trntstmode+'_Y']

            # For printing values, we use the full trn/val/tst batch. However, this is too large (in some cases)
            #   for calculating the nonlinearIB MI estimator, which creates a N^2 matrix. For this reason, we 
            #   estimate the IB values using a random subsample of 2000 points
            v = sess.run({'ce': net.cross_entropy, 'acc': net.accuracy, 'Iyt': Iyt}, 
                         feed_dict={'X:0': X, 'trueY:0': Y, 'entropyY:0': data[trntstmode+'_entropyY']})
            
            ix = np.random.choice(len(X), 2000)
            v.update(sess.run({'Ixt': lobj.Ixt, 'Ixt_lb': lobj.Ixt_lb, 'vIxt': lobj.vIxt, 'mi_penalty': mi_penalty}, 
                              feed_dict={'X:0': X[ix]}))
            v['loss'] = v['mi_penalty'] - v['Iyt']
            cdata[trntstmode] = v
                  
        if do_print:
            print()
            time_per_epoch = '-'
            if epoch > 0 and start_time is not None:
                time_per_epoch = '%0.3f' % ((time.time() - start_time) / epoch)
            
            print('method: {method} epoch: {epoch:d} | beta: {beta:0.4f} | noisevar: {noisevar:g} | time/epoch: {t}'.
                  format(t=time_per_epoch, **cdata))
            for mlist in [['ce','acc', 'loss'], ['Ixt', 'Ixt_lb', 'vIxt', 'Iyt']]:
                for m in mlist:
                    print("%s: % 0.3f/% 0.3f/% 0.3f | " % (m, cdata['trn'][m], cdata['val'][m], cdata['tst'][m]), end="")
                print()
                
        return cdata
    
    
    print("*** Saving to %s ***" % savedir)
    print("CONFIGURATION:", cfg)
    
    saver = tf.train.Saver(max_to_keep=cfg['patience']+5)
    
    if method in ['nlIB','VIB']:
        mi_power = (2 if cfg['objective']=='sq' else 1)
        mi_penalty = beta*(net.iblayerobj.Ixt if method=='nlIB' else net.iblayerobj.vIxt)**mi_power
    else:
        mi_penalty = tf.constant(0.0)
    
    saved_data = []
    Iyt        = net.entropyY - net.cross_entropy
    loss       = mi_penalty - Iyt
    
    var_list = None
    if method == 'ce': # minimize cross-entropy only, do not change noise variance parameter
        var_list = [v for v in tf.trainable_variables() if v not in [net.iblayerobj.phi]]

    optimizer  = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) 
    trainstep  = optimizer.minimize(loss, var_list=var_list)
    
    sess.run(tf.variables_initializer(optimizer.variables()))
    sess.run(net.iblayerobj.phi.assign(utils.softplusinverse(cfg['initnoise'])))

    start_time = time.time()
    
    cdata = calcstats(epoch=0, do_print=True)
    saved_data.append(cdata)
    utils.write_data(savedir + '/data', [cfg, saved_data])
    
    last_val_increase_epoch = 0
    last_best_val           = None
    for epoch in range(1,1000): # For convenience, we start epoch numbering at 1, not 0
        # also, 1000 max epochs is just a number -- earlystopping will typically kick in before this is reached

        if (epoch - last_val_increase_epoch) >= cfg['patience']:
            print("Validation hasn't improved since epoch %d (in %d epochs) -- quiting" % 
                  (last_val_increase_epoch, cfg['patience']))
            break
            
        batches = utils.get_train_batches(data['trn_X'], data['trn_Y'], cfg['batchsize'])
        for batchndx, batch in enumerate(batches):
            sess.run(trainstep, feed_dict=batch)
        
        cdata = calcstats(epoch, (epoch-1) % cfg['report_every'] == 0)
        saved_data.append(cdata)
        utils.write_data(savedir + '/data', [cfg, saved_data])
        
        if last_best_val is None or cdata['val']['loss'] <= last_best_val:
            saver.save(sess, savedir + '/tf_model', global_step=epoch)
            last_val_increase_epoch = epoch
            last_best_val = cdata['val']['loss']

    # Cleanup
    print("Cleaning up...")
    for fname in os.listdir(savedir):
        if fname.startswith('tf_model-') and not fname.startswith('tf_model-%d.'%last_val_increase_epoch):
            delfname = os.path.join(savedir, fname)
            os.remove(delfname)
        
    print("Model saved in path:", savedir)
    return saved_data


class Network(object):
    def __init__(self, input_dim, output_dim, init_noisevar=None):
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        self.true_outputs   = tf.placeholder(tf.float32, [None,output_dim,], name='trueY')
        
        self.iblayerobj     = iblayer.NoisyIBLayer(init_noisevar=init_noisevar, name='noisy_ib_layer')
        
        self.layers = []
        self.layers.append( tf.placeholder(tf.float32, [None,input_dim,], name='X') )
        self.layers.append( tf.keras.layers.Dense(cfg['n_wide'], activation=tf.nn.relu)(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(cfg['n_wide'], activation=tf.nn.relu)(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(cfg['n_hidden'], activation=None)(self.layers[-1]) )
        self.layers.append( self.iblayerobj(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(cfg['n_wide'], activation=tf.nn.relu)(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(output_dim, activation=None, name='Y')(self.layers[-1]) )

        self.inputs         = self.layers[0]
        self.predictions    = self.layers[-1]

        self.entropyY       = tf.placeholder(dtype=tf.float32, shape=(), name='entropyY')
        
        self.cross_entropy  = utils.get_error    (errtype=data['err'], y_true=self.true_outputs, y_pred=self.predictions)
        self.accuracy       = utils.get_accuracy (errtype=data['err'], y_true=self.true_outputs, y_pred=self.predictions)
        
        
        


if 'ce' in run_methods:
    # Train the base model, without compression penalty
    tf.reset_default_graph()
    with tf.Session(config=tfconfig) as sess:
        print("Making baseline model")
        n = Network(input_dim, output_dim, init_noisevar=1e-15)
        sess.run(tf.global_variables_initializer())
        train(sess, method='ce', beta=0.0, net=n, cfg=cfg, data=data, savedir=savedir+'/basemodel')
        del n

    
for runndx in range(cfg['n_runs']):
    for beta in betavals:
        if np.isclose(beta, 0): continue
        
        for method in run_methods:
            if method == 'ce' : 
                continue 
            elif method not in ['VIB','nlIB']:
                raise Exception("Unknown method %s" % method)
                
            tf.reset_default_graph()
            with tf.Session(config=tfconfig) as sess:
                n = Network(input_dim, output_dim)
                sess.run(tf.global_variables_initializer())
                
                savename = savedir + '/results-%s-%0.5f-%s-run%d' % (method, beta, cfg['objective'], runndx)
                
                train(sess, method=method, beta=beta, net=n, cfg=cfg, data=data, savedir=savename)
                
                del n
                print("\n")

            
            
           