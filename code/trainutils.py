import numpy as np
import os, pickle
import tensorflow as tf

def stats(sess, mode, beta, loss, epoch, data, n, do_print=False):
    noisevar, etavar = sess.run([n.noisevar, n.etavar])

    measures = {'ce': n.cross_entropy, 'loss': loss, 'acc': n.accuracy, 
                # 'activations': n.encoder[-1],
                'Ixt': n.Ixt, 'Ixt_lb': n.Ixt_lb, 'vIxt': n.vIxt, 'Iyt': n.Iyt}
    
    cdata = {'mode': mode, 'epoch': epoch, 'beta': beta, 'noisevar': noisevar, 'etavar': etavar} 
    
    for r in ['trn', 'tst']:
        permutation = np.random.permutation(len(data['trn_Y']))
        d = data['trn_X'][permutation]
        l = data['trn_Y'][permutation]

        m = sess.run(measures, feed_dict={n.x: d[:2000], n.y: l[:2000]})
        cdata[r] = {}
        for k, v in m.items():
            cdata[r][k] = v
            
    if do_print:
        print()
        print('mode: %s epoch: %d | beta: %0.4f | noisevar: %g | kw: %g' % (mode, epoch+1, beta, noisevar, etavar))
        for mlist in [['ce','acc', 'loss'], ['Ixt', 'Ixt_lb', 'vIxt', 'Iyt']]:
            for m in mlist:
                print("%s: % 0.3f/% 0.3f | " % (m, cdata['trn'][m], cdata['tst'][m]), end="")
            print()
    return cdata

def write_data(savedir, epoch, sess, saver, data):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    with open(savedir + '/data', 'wb') as fp:
        pickle.dump(data, fp)
    saver.save(sess, savedir + '/tf_model', global_step=epoch)


def train(sess, saver, mode, beta, cfg, data, n, optimizer, report_every, savedir, fit_var=False):
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
    
    print("*** Saving to %s ***" % savedir)
    
    n_batch          = cfg['n_batch']
    n_noisevar_batch = cfg['n_noisevar_batch']
    saved_data       = []
    n_mini_batches   = int(len(data['trn_Y']) / n_batch)
    

    var_list = None
    if mode == 'ce': # minimize cross-entropy only, do not change eta and phi parameters
        loss = n.cross_entropy
        var_list = n.no_var_params

    elif mode in ['nlIB', 'VIB']:
        Ixt  = n.Ixt if mode == 'nlIB' else n.vIxt
        loss = beta*(Ixt**2 if cfg['squaredIB'] else Ixt) - n.Iyt

    else:
        raise Exception('Unknown mode')  


    trainstep = optimizer.minimize(loss, var_list=var_list)
    sess.run(tf.variables_initializer(optimizer.variables()))

    cdata = stats(sess, mode, beta, loss, 0, data, n, do_print=False)
    saved_data.append(cdata)
    write_data(savedir, 0, sess, saver, [cfg, saved_data])

    for epoch in range(cfg['n_epochs']):
        # randomize order of training data
        permutation  = np.random.permutation(len(data['trn_Y']))
        train_X = data['trn_X'][permutation]
        train_Y = data['trn_Y'][permutation]

        x_batch, y_batch, dmatrix = None, None, None

        if mode != 'ce':
            # Set kernel width
            x_batch = train_X[:n_noisevar_batch]
            y_batch = train_Y[:n_noisevar_batch]
            dmatrix = sess.run(n.distance_matrix, feed_dict={n.x: x_batch})
            n.eta_optimizer.minimize(sess, feed_dict={n.distance_matrix_ph: dmatrix})

            # Set noise variance with scipy, if needed
            if (fit_var and epoch == 0) or (cfg['train_noisevar']=='scipy' and epoch % 30 == 0):
                opt = tf.contrib.opt.ScipyOptimizerInterface(loss, var_list=[n.phi])
                opt.minimize(sess, feed_dict={n.x: x_batch, n.y: y_batch})


        for batch in range(n_mini_batches):
            # sample mini-batch
            x_batch = train_X[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_Y[batch * n_batch:(1 + batch) * n_batch]
            cparams = {n.x: x_batch, n.y: y_batch}
            sess.run(trainstep, feed_dict=cparams)

        cdata = stats(sess, mode, beta, loss, epoch+1, data, n, do_print=epoch % report_every == 0)
        saved_data.append(cdata)
        write_data(savedir, epoch+1, sess, saver, [cfg, saved_data])

    return saved_data