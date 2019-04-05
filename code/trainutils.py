import numpy as np
import os, pickle, time
import tensorflow as tf

def stats(sess, mode, beta, loss, epoch, data, n, start_time=None, do_print=False):
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
        if epoch > 0 and start_time is not None:
            time_per_epoch = '%0.3f' % ((time.time() - start_time) / epoch)
        else:
            time_per_epoch = '-'
            
        print('mode: %s epoch: %d | beta: %0.4f | noisevar: %g | kw: %g | time/epoch: %s' 
               % (mode, epoch, beta, noisevar, etavar, time_per_epoch))
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

    trainstep  = optimizer.minimize(loss, var_list=var_list)
    sess.run(tf.variables_initializer(optimizer.variables()))

    cdata = stats(sess, mode, beta, loss, 0, data, n, do_print=True)
    saved_data.append(cdata)
    write_data(savedir, 0, sess, saver, [cfg, saved_data])

    noisevar_optimizer = None
    
    start_time = time.time()
    for epoch in range(cfg['n_epochs']):
        # randomize order of training data
        permutation  = np.random.permutation(len(data['trn_Y']))
        train_X = data['trn_X'][permutation]
        train_Y = data['trn_Y'][permutation]

        x_batch, y_batch, dmatrix = None, None, None

        if mode != 'ce':
            
            train_kdewidth = cfg['train_kdewidth']
            train_noisevar = (fit_var and epoch == 0) or (cfg['train_noisevar']=='optimizer') #  and epoch % 30 == 0)
            
            if train_kdewidth or train_noisevar:
                x_batch = train_X[:n_noisevar_batch]
                y_batch = train_Y[:n_noisevar_batch]
                #dmatrix = sess.run(n.distance_matrix, feed_dict={n.x: x_batch})
            
            if train_kdewidth: # Set kernel width
                n.eta_optimizer.minimize(sess, feed_dict={n.x: x_batch}) # feed_dict={n.distance_matrix: dmatrix})

            if train_noisevar: # Set noise variance with an external optimizer
                if noisevar_optimizer is None:
                    noisevar_optimizer = n.get_noisevar_optimizer(loss)
                noisevar_optimizer.minimize(sess, feed_dict={n.x: x_batch, n.y: y_batch})# , n.distance_matrix: dmatrix})


        for batch in range(n_mini_batches):
            # sample mini-batch
            x_batch = train_X[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_Y[batch * n_batch:(1 + batch) * n_batch]
            cparams = {n.x: x_batch, n.y: y_batch}
            sess.run(trainstep, feed_dict=cparams)

        cdata = stats(sess, mode, beta, loss, epoch+1, data, n, start_time=start_time, do_print=epoch % report_every == 0)
        saved_data.append(cdata)
        write_data(savedir, epoch+1, sess, saver, [cfg, saved_data])

    return saved_data