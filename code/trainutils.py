import time, os, pickle
import numpy as np
import tensorflow as tf
import signal
import logging

class DelayedKeyboardInterrupt(object):
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug('SIGINT received. Delaying KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


def train(sess, mode, beta, cfg, data, net, savedir, optimization_callback=None, fit_var=False):
    # TODO : document
    # sess         : TensorFlow session
    # mode         : 'ce' (cross-entropy), 'nlIB' (MoG estimator), or 'VIB' (variational IB)
    # beta         : beta value
    # cfg          : configuration dictionary
    # data         : data object
    # n            : neural network object
    # optimizer    : optimizer object
    # report_every : how often to spit out stats
    # fname        : filename where to save results
    # fit_var      : whether to initially fit noisevariance or not

    def write_data(epoch, saveobjs):
        with DelayedKeyboardInterrupt():
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            with open(savedir + '/data', 'wb') as fp:
                pickle.dump(saveobjs, fp)
            saver.save(sess, savedir + '/tf_model', global_step=epoch)

    
    def report(epoch, do_print=False):
        noisevar, etavar = sess.run([net.iblayerobj.noisevar, net.iblayerobj.etavar])

        measures = {'ce': net.cross_entropy, 'loss': loss, 'acc': net.accuracy, 
                    # 'activations': n.encoder[-1],
                    'Ixt': net.iblayerobj.Ixt, 'Ixt_lb': net.iblayerobj.Ixt_lb, 
                    'vIxt': net.iblayerobj.vIxt, 'Iyt': Iyt}

        cdata = {'mode': mode, 'epoch': epoch, 'beta': beta, 'noisevar': noisevar, 'etavar': etavar} 

        for r in ['trn', 'tst']:
            permutation = np.random.permutation(len(data['trn_Y']))
            d = data['trn_X'][permutation]
            l = data['trn_Y'][permutation]

            m = sess.run(measures, feed_dict={net.inputs: d[:2000], net.true_outputs: l[:2000]})
            cdata[r] = {}
            for k, v in m.items():
                cdata[r][k] = v

        if do_print:
            print()
            time_per_epoch = '-'
            if epoch > 0 and start_time is not None:
                time_per_epoch = '%0.3f' % ((time.time() - start_time) / epoch)
            
            print('mode: %s epoch: %d | beta: %0.4f | noisevar: %g | kw: %g | time/epoch: %s' 
                   % (mode, epoch, beta, noisevar, etavar, time_per_epoch))
            for mlist in [['ce','acc', 'loss'], ['Ixt', 'Ixt_lb', 'vIxt', 'Iyt']]:
                for m in mlist:
                    print("%s: % 0.3f/% 0.3f | " % (m, cdata['trn'][m], cdata['tst'][m]), end="")
                print()
        return cdata
    
    print("*** Saving to %s ***" % savedir)
    saver = tf.train.Saver(max_to_keep=30)
    
    optimizer      = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) 
    
    n_batch          = cfg['n_batch']
    n_noisevar_batch = cfg['n_noisevar_batch']
    saved_data       = []
    n_mini_batches   = int(len(data['trn_Y']) / n_batch)
    

    var_list = None
    Iyt      = data['entropyY'] - net.cross_entropy
    if mode == 'ce': # minimize cross-entropy only, do not change eta and phi parameters
        loss = net.cross_entropy
        var_list = [v for v in tf.trainable_variables() if v not in [net.iblayerobj.phi, net.iblayerobj.eta]]

    elif mode in ['nlIB', 'VIB']:
        Ixt  = net.iblayerobj.Ixt if mode == 'nlIB' else net.iblayerobj.vIxt
        loss = beta*(Ixt**2 if cfg['squaredIB'] else Ixt) - Iyt

    else:
        raise Exception('Unknown mode %s' % mode)  


    trainstep  = optimizer.minimize(loss, var_list=var_list)
    sess.run(tf.variables_initializer(optimizer.variables()))

    cdata = report(epoch=0, do_print=True)
    saved_data.append(cdata)
    write_data(0, [cfg, saved_data])

    start_time = time.time()
    
    for epoch in range(cfg['n_epochs']):
        # randomize order of training data
        permutation  = np.random.permutation(len(data['trn_Y']))
        train_X = data['trn_X'][permutation]
        train_Y = data['trn_Y'][permutation]

        if optimization_callback is not None:
            optimization_callback(sess, net.inputs, train_X, net.true_outputs, train_Y)

        for batch in range(n_mini_batches): # sample mini-batch
            x_batch = train_X[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_Y[batch * n_batch:(1 + batch) * n_batch]
            cparams = {net.inputs: x_batch, net.true_outputs: y_batch}
            sess.run(trainstep, feed_dict=cparams)

        cdata = report(epoch+1, epoch % cfg['report_every'] == 0)
        saved_data.append(cdata)
        write_data(epoch+1, [cfg, saved_data])
        
        
    return saved_data
