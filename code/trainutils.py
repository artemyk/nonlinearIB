import numpy as np
import os, pickle
import tensorflow as tf

def stats(sess, epoch, data, n, do_print=False):
    beta, noisevar, etavar = sess.run([n.beta, n.noisevar, n.etavar])

    measures = {'ce': n.cross_entropy, 'acc': n.accuracy, 'activations': n.encoder[-1],
                'nlIBloss': n.nlIB_loss.f, 'VIBloss': n.VIB_loss.f, 
                'Ixt': n.Ixt, 'Ixt_lb': n.Ixt_lb, 'vIxt': n.vIxt, 'Iyt': n.Iyt}
    
    cdata = {'epoch': epoch, 'beta': beta, 'noisevar': noisevar, 'etavar': etavar} 
    for r in ['trn', 'tst']:
        permutation = np.random.permutation(len(data['trn_labels']))
        d = data['trn_data'][permutation]
        l = data['trn_labels'][permutation]

        m = sess.run(measures, feed_dict={n.x: d[:2000], n.y: l[:2000]})
        cdata[r] = {}
        for k, v in m.items():
            cdata[r][k] = v
            
    if do_print:
        print()
        print('epoch: %d | beta: %0.4f | noisevar: %g | kw: %g' % (epoch+1, beta, noisevar, etavar))
        for mlist in [['ce','acc', 'nlIBloss', 'VIBloss'], ['Ixt', 'Ixt_lb', 'vIxt', 'Iyt']]:
            for m in mlist:
                print("%s: % 0.2f / % 0.2f | " % (m, cdata['trn'][m], cdata['tst'][m]), end="")
            print()
    return cdata

def write_data(d, fname):
    fdir = os.path.dirname(fname)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(fname, 'wb') as fp:
        pickle.dump(d, fp)



def train(sess, cfg, data, n, loss, report_every, fname, fit_var=False):
    n_batch = cfg['n_batch']
    beta    = cfg['beta']
    sess.run(n.beta.assign(beta))

    n_mini_batches = int(len(data['trn_labels']) / n_batch)
    saved_data = []

    for epoch in range(cfg['n_epochs']):
        # randomize order of training data
        permutation  = np.random.permutation(len(data['trn_labels']))
        train_data   = data['trn_data'][permutation]
        train_labels = data['trn_labels'][permutation]
        
        if (fit_var and epoch == 0) or not cfg['gradient_train_noisevar']:
            x_batch = train_data[:cfg['n_noisevar_batch']]
            y_batch = train_labels[:cfg['n_noisevar_batch']]
            opt = tf.contrib.opt.ScipyOptimizerInterface(loss.f, var_list=[n.phi])
            opt.minimize(sess, feed_dict={n.x: x_batch, n.y: y_batch})
        if True:
            x_batch = train_data[:cfg['n_noisevar_batch']]
            y_batch = train_labels[:cfg['n_noisevar_batch']]
            dmatrix = sess.run(n.distance_matrix, feed_dict={n.x: x_batch})
            n.eta_optimizer.minimize(sess, feed_dict={n.distance_matrix_ph: dmatrix})
            
        cdata = stats(sess, epoch, data, n, epoch % report_every == 0)
        saved_data.append(cdata)
        write_data([cfg, saved_data], fname)

        for batch in range(n_mini_batches):
            # sample mini-batch
            x_batch = train_data[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_labels[batch * n_batch:(1 + batch) * n_batch]

            cparams = {n.x: x_batch, n.y: y_batch}
            sess.run(loss.trainstep, feed_dict=cparams)

    cdata = stats(sess, epoch+1, data, n, do_print=True)
    saved_data.append(cdata)
    write_data([cfg, saved_data], fname)

    return saved_data