import numpy as np
import os, pickle
import tensorflow as tf

def load_mnist(n_data=None):
    (train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

    # randomize order
    permutation = np.random.permutation(len(train_labels))
    train_data = train_data[permutation]
    train_labels = train_labels[permutation]
    permutation = np.random.permutation(len(test_labels))
    test_data = test_data[permutation]
    test_labels = test_labels[permutation]

    # normalize, reshape, and convert to one-hot vectors
    train_data = np.reshape(train_data, (-1, 784)) / (255./2.) - 1.
    test_data = np.reshape(test_data, (-1, 784)) / (255./2.) - 1.
    train_labels = one_hot(train_labels)
    test_labels = one_hot(test_labels)

    if n_data is not None:
        data = {'trn_X': train_data[:n_data], 'trn_Y': train_labels[:n_data], 
                'tst_X': test_data[:n_data] , 'tst_Y': test_labels[:n_data]}
    else:
        data = {'trn_X': train_data, 'trn_Y': train_labels, 
                'tst_X': test_data , 'tst_Y': test_labels}

    data['entropyY'] = np.log(10)
    return data

def load_wine():
    mx = np.vstack([
        np.genfromtxt('data/winequality-red.csv',delimiter=";", skip_header=1),
        np.genfromtxt('data/winequality-white.csv',delimiter=";", skip_header=1),
    ])
    np.random.seed(12345)
    permutation  = np.random.permutation(len(mx))
    mx = mx[permutation,:]

    X = mx[permutation,:-1]
    y = mx[permutation,-1]
    #Y = one_hot(y.astype('int')) # 
    Y = np.zeros( (len(mx), 2))
    Y[y < 6,0] = 1.0 
    Y[y >= 6,1] = 1.0 
    #Y[y == 5,:] = 0.5
    ps = Y.mean(axis=0)
    entropyY = np.sum([-p*np.log(p) for p in ps if not np.isclose(p,0)])
                      
    hl = int(len(mx)/2)

    data = { 'trn_X' : X[:hl,:], 'trn_Y': Y[:hl,:],
             'tst_X' : X[hl:,:], 'tst_Y': Y[hl:,:],
             'entropyY' : entropyY}
    
    return data

def load_szt():
    # Data from artificial dataset used in Schwartz-Ziv and Tishby
    d1 = scipy.io.loadmat('data/g1.mat')
    d2 = scipy.io.loadmat('data/g2.mat')
    data = { 'trn_X' : d1['F'].astype('float32'), 'trn_Y': trainutils.one_hot(d1['y'].flat),
             'tst_X' : d2['F'].astype('float32'), 'tst_Y': trainutils.one_hot(d2['y'].flat),
             'entropyY': np.log(2)}
    return data
    

def one_hot(x, n_classes=None):
    assert(np.array(x).ndim == 1)
    
    # input: 1D array of N labels, output: N x max(x)+1 array of one-hot vectors
    if n_classes is None:
        n_classes = max(x) + 1

    x_one_hot = np.zeros([len(x), n_classes])
    x_one_hot[np.arange(len(x)), x] = 1
    return x_one_hot


def stats(sess, mode, beta, loss, epoch, data, n, do_print=False):
    noisevar, etavar = sess.run([n.noisevar, n.etavar])

    measures = {'ce': n.cross_entropy, 'acc': n.accuracy, 'activations': n.encoder[-1],
                'loss': loss, 'Ixt': n.Ixt, 'Ixt_lb': n.Ixt_lb, 'vIxt': n.vIxt, 'Iyt': n.Iyt}
    
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

def write_data(d, fname):
    fdir = os.path.dirname(fname)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    with open(fname, 'wb') as fp:
        pickle.dump(d, fp)



def train(sess, mode, beta, cfg, data, n, optimizer, report_every, fname, fit_var=False):
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
        
        
        cdata = stats(sess, mode, beta, loss, epoch, data, n, epoch % report_every == 0)
        saved_data.append(cdata)
        write_data([cfg, saved_data], fname)

        for batch in range(n_mini_batches):
            # sample mini-batch
            x_batch = train_X[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_Y[batch * n_batch:(1 + batch) * n_batch]
            cparams = {n.x: x_batch, n.y: y_batch}
            sess.run(trainstep, feed_dict=cparams)

    cdata = stats(sess, mode, beta, loss, epoch+1, data, n, do_print=True)
    saved_data.append(cdata)
    write_data([cfg, saved_data], fname)

    return saved_data