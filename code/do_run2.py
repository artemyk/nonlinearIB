import numpy as np
import matplotlib.pyplot as plt
import pathlib, os, pickle, time, argparse

cfg = {}
cfg['n_batch']           = 128
cfg['n_noisevar_batch']  = 2000 
cfg['report_every']      = 10
cfg['n_wide']            = 128
cfg['encoder_layers']    = 2

do_pretrain  = False
max_epochs   = 1000  # max epochs (earlystopping will typically kick in much earlier)
betavals     = 10**np.linspace(-4, 0.1, 20, endpoint=True) 


parser = argparse.ArgumentParser()
parser.add_argument("-squaredIB", help="Use I(Y:T) - beta*I(X:T)^2 objective function?", action="store_true")
parser.add_argument("-n_runs"   , help="How many runs to do", default=1, type=int)
parser.add_argument("-n_hidden" , help="How many neurons in bottleneck layer", default=5, type=int)
parser.add_argument("-early_stopping_patience", default=10, type=int,
                                  help="How many epochs before validation improvement for early stoppoing")
parser.add_argument('runtype',    help="Which dataset", choices=['MNIST','FashionMNIST','Housing','Wine'])
parser.add_argument('outputdir',  help="Where to store output files", type=str)

args = parser.parse_args()
for k in ['squaredIB', 'n_runs', 'n_hidden', 'early_stopping_patience','runtype']: 
    cfg[k] = getattr(args, k)

savedirbase  = str(pathlib.Path().absolute()) + '/' + args.outputdir + '/'
savedir = savedirbase + cfg['runtype'] 

if not os.path.exists(savedir):
    print("Making directory", savedir)
    os.makedirs(savedir)

    
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth=True

import loaddata, iblayer, utils
from trainutils import error


def train(sess, mode, beta, n_epochs, cfg, data, net, savedir):
    # sess         : TensorFlow session
    # mode         : 'ce' (cross-entropy only), 'nlIB' (MoG estimator), or 'VIB' (variational IB)
    # beta         : beta value
    # cfg          : configuration dictionary
    # data         : data object
    # net          : neural network object
    # savedir      : directory where to save results

    def get_loss(ce, pIxt, pIyt):
        if mode == 'ce':
            return ce
        else:
            return beta*(pIxt**2 if cfg['squaredIB'] else pIxt) - pIyt
        
    def write_data(epoch, saveobjs):
        with utils.DelayedKeyboardInterrupt():
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            with open(savedir + '/data', 'wb') as fp:
                pickle.dump(saveobjs, fp)
            saver.save(sess, savedir + '/tf_model', global_step=epoch)

    
    def report(epoch, do_print=False):
        lobj     = net.iblayerobj
        noisevar = sess.run(lobj.noisevar)
        cdata    = {'mode': mode, 'epoch': epoch, 'beta': beta, 'noisevar': noisevar} 

        for trntstmode in ['trn', 'val', 'tst']:
            X, Y = data[trntstmode+'_X'], data[trntstmode+'_Y']

            m = sess.run({'ce': net.cross_entropy, 'acc': net.accuracy, 'Iyt': Iyt}, 
                         feed_dict={net.inputs: X, net.true_outputs: Y})
            
            cdata[trntstmode] = {}
            for k, v in m.items():
                cdata[trntstmode][k] = v
            permutation = np.random.permutation(len(X))
            m = sess.run({'Ixt': lobj.Ixt, 'Ixt_lb': lobj.Ixt_lb, 'vIxt': lobj.vIxt, }, 
                         feed_dict={net.inputs: X[permutation[:2000]]})
            for k, v in m.items():
                cdata[trntstmode][k] = v
                
            cdata[trntstmode]['loss'] = get_loss(cdata[trntstmode]['ce'], cdata[trntstmode]['Ixt'], cdata[trntstmode]['Iyt'])
            
        if do_print:
            print()
            time_per_epoch = '-'
            if epoch > 0 and start_time is not None:
                time_per_epoch = '%0.3f' % ((time.time() - start_time) / epoch)
            
            print('mode: %s epoch: %d | beta: %0.4f | noisevar: %g | time/epoch: %s' 
                   % (mode, epoch, beta, noisevar, time_per_epoch))
            for mlist in [['ce','acc', 'loss'], ['Ixt', 'Ixt_lb', 'vIxt', 'Iyt']]:
                for m in mlist:
                    print("%s: % 0.3f/% 0.3f/% 0.3f | " % (m, cdata['trn'][m], cdata['val'][m], cdata['tst'][m]), end="")
                print()
        return cdata
    
    
    print("*** Saving to %s ***" % savedir)
    saver = tf.train.Saver(max_to_keep=30)
    
    optimizer      = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) 
    
    n_batch          = cfg['n_batch']
    n_noisevar_batch = cfg['n_noisevar_batch']
    saved_data       = []
    n_mini_batches   = int(np.ceil(len(data['trn_Y']) / n_batch))
    

    var_list = None
    Iyt      = data['trn_entropyY'] - net.cross_entropy
    
    Ixt  = net.iblayerobj.Ixt if mode != 'VIB' else net.iblayerobj.vIxt
    loss = get_loss(net.cross_entropy, Ixt, Iyt)
    
    if mode == 'ce': # minimize cross-entropy only, do not change noise variance parameter
        var_list = [v for v in tf.trainable_variables() if v not in [net.iblayerobj.phi]]

    trainstep  = optimizer.minimize(loss, var_list=var_list)
    sess.run(tf.variables_initializer(optimizer.variables()))

    cdata = report(epoch=0, do_print=True)
    saved_data.append(cdata)
    write_data(0, [cfg, saved_data])
    
    start_time = time.time()
    
    permutation  = np.random.permutation(len(data['trn_Y']))
    train_X = data['trn_X'][permutation]
    train_Y = data['trn_Y'][permutation]
    if mode in ['VIB', 'nlIB']:
        # DOCUMENT
        cparams = {net.inputs: train_X[0:100], net.true_outputs: train_Y[0:100]}
        #noise = sess.run(net.iblayerobj.noise, feed_dict=cparams)
        #cparams_withnoise = {net.inputs: x_batch, net.true_outputs: y_batch, net.iblayerobj.noise: noise}
        dist_matrix = sess.run(net.iblayerobj.dist_matrix, feed_dict=cparams)
        max_dist    = np.sqrt(dist_matrix.max())
        if np.isnan(max_dist) or np.isclose(max_dist, 0):
            max_dist = 0.01

        x_batch = train_X[0:cfg['n_noisevar_batch']]
        y_batch = train_Y[0:cfg['n_noisevar_batch']]
        noise = sess.run(net.iblayerobj.noise, feed_dict={net.inputs: x_batch, net.true_outputs: y_batch})
        minloss, bestphi = None, None
        for noisevar in np.geomspace(1e-10, max_dist, 100):
            cur_phi   = utils.softplusinverse(noisevar)
            feed_dict = {net.inputs: x_batch, net.true_outputs: y_batch, 
                         net.iblayerobj.noise: noise, net.iblayerobj.phi: cur_phi}
            lossval   = sess.run(loss, feed_dict=feed_dict)

            lossval   = np.round(lossval, 2) # !DOCUMENT! for stability

            #!DOCUMENT! note the lossval <= minloss --- we choose largest allowable
            if not np.isnan(lossval) and (minloss is None or lossval <= minloss):
                minloss = lossval
                bestphi = cur_phi

        sess.run(net.iblayerobj.phi.assign(bestphi))
        
    last_val_increase_epoch = 0
    last_best_val           = None

    for epoch in range(n_epochs):
        if (epoch - last_val_increase_epoch) >= cfg['early_stopping_patience']:
            print("Validation hasn't improved in %s epochs -- quiting" % cfg['early_stopping_patience'])
            break
            
        # randomize order of training data
        permutation  = np.random.permutation(len(data['trn_Y']))
        train_X = data['trn_X'][permutation]
        train_Y = data['trn_Y'][permutation]

        for batch in range(n_mini_batches): # sample mini-batch
            x_batch = train_X[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_Y[batch * n_batch:(1 + batch) * n_batch]
            cparams = {net.inputs: x_batch, net.true_outputs: y_batch}
            sess.run(trainstep, feed_dict=cparams)
        
        cdata = report(epoch+1, epoch % cfg['report_every'] == 0)
        saved_data.append(cdata)
        write_data(epoch+1, [cfg, saved_data])
        
        if last_best_val is None or cdata['val']['loss'] <= last_best_val:
            last_val_increase_epoch = epoch
            last_best_val = cdata['val']['loss']
        
    return saved_data


class Network(object):
    def __init__(self, input_dim, output_dim):
        self.input_dim      = input_dim
        self.output_dim     = output_dim
        
        self.iblayerobj     = iblayer.NoisyIBLayer(init_noisevar=1e-15)  # , init_kdewidth=-20)
        
        self.true_outputs   = tf.placeholder(tf.float32, [None,output_dim,], name='true_outputs')

        # TODO: build the network
        self.layers = []
        self.layers.append( tf.placeholder(tf.float32, [None,input_dim,], name='X') )
        for _ in range(cfg['encoder_layers']):
            self.layers.append( tf.keras.layers.Dense(cfg['n_wide'], activation=tf.nn.relu)(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(cfg['n_hidden'], activation=None)(self.layers[-1]) )
        self.layers.append( self.iblayerobj(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(cfg['n_wide'], activation=tf.nn.relu)(self.layers[-1]) )
        self.layers.append( tf.keras.layers.Dense(output_dim, activation=None)(self.layers[-1]) )

        self.inputs         = self.layers[0]
        self.predictions    = self.layers[-1]

        f                   = error(errtype=data['err'], y_true=self.true_outputs, y_pred=self.predictions)
        self.cross_entropy  = tf.reduce_mean(f) # cross entropy
        self.accuracy       = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.predictions, 1), 
                                                              tf.argmax(self.true_outputs, 1)), tf.float32))
        
        
        
data           = loaddata.load_data(cfg['runtype'], validation=True)
input_dim      = data['trn_X'].shape[1]
output_dim     = data['trn_Y'].shape[1]

if do_pretrain or not os.path.exists(savedir+'/basemodel'):
    if not os.path.exists(savedir+'/basemodel'):
        print('%s doesnt exist --- creating basemodel' % (savedir+'/basemodel'))
    # Train the base model, without compression
    tf.reset_default_graph()
    with tf.Session(config=tfconfig) as sess:
        print("Making base model")
        n = Network(input_dim, output_dim)
        sess.run(tf.global_variables_initializer())
        train(sess, mode='ce', beta=0.0, n_epochs=max_epochs,
              net=n, cfg=cfg, data=data, savedir=savedir+'/basemodel')
        print("Model saved in path: %s" % savedir)
        del n

    
for runndx in range(cfg['n_runs']):
    for beta in betavals:
        if np.isclose(beta, 0):
            continue
        for mode in ['nlIB', 'VIB']:
            tf.reset_default_graph()
            with tf.Session(config=tfconfig) as sess:
                n = Network(input_dim, output_dim)

                sess.run(tf.global_variables_initializer())
                tf.train.Saver().restore(sess, tf.train.latest_checkpoint(savedir+'/basemodel'))

                sqmode = 'sq'  if cfg['squaredIB'] else 'reg'
                savename = savedir + '/results-%s-%0.5f-%s-run%d' % (mode, beta, sqmode, runndx)
                train(sess, mode=mode, beta=beta, n_epochs=max_epochs, net=n, cfg=cfg, data=data, savedir=savename)
                print("Model saved in path: %s" % savedir)
                del n
                print()
                print()



            
            
           