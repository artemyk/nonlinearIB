import numpy as np
import scipy
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import time, os, pickle, pathlib

import model
import trainutils
import loaddata

report_every = 10  # how often to print stats during training
n_runs       = 1   # how many times to repeat the whole scan across beta's
savedirbase  = str(pathlib.Path().absolute()) + '/saveddata-scipy/'


# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.01
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                           100000, 0.96, staircase=True)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) 

runtype = 'MNIST'

cfg = {
    'runtype'         : runtype,      # which dataset we're running
    'n_batch'         : 128 ,         # SGD batch size
    'train_noisevar'  : 'gradient',   # train noise variance with gradient descent ('gradient'), 
                                      #  an external optimizer loop ('optimizer'), or leave fixed ('none')
    'n_noisevar_batch': 1000,         # batch size for training noise variance when train_noisevar='optimizer'
    'initial_fitvar'  : False,        # whether to set noisevar to optimal value before training
    'squaredIB'       : False,        # optimize I(Y;T)-beta*I(X;T) or I(Y;T)-beta*I(X;T)^2 
    'err_func'        : 'softmax_ce', # 'softmax_ce' for classification, 'mse' for regression  
    'train_kdewidth'  : True,         # whether to adapt the kernel width 
}
cfg['train_noisevar'] = 'optimizer'
cfg['train_kdewidth'] = False


betavals = None
data     = loaddata.load_data('MNIST')

savedir = 'MNIST/demo'
cfg.update({
    'n_epochs'    : 150,
    'squaredIB'   : True,
    'encoder_arch': [(512,tf.nn.relu),(512,tf.nn.relu),(2,None)], 
    'decoder_arch': [(512,tf.nn.relu),(10,None)],
})

savedir = savedirbase + savedir
cfg['optimizer'] = repr(optimizer)

betavals = 10**np.linspace(-5, 0.1, 30, endpoint=True)



def get_net():


for beta in betavals:
    if np.isclose(beta,0): 
        continue
    for mode in ['nlIB','VIB',]:
        tf.reset_default_graph()
        with tf.Session() as sess:

            
            n = model.Net(input_dims   = data['trn_X'].shape[1],
                          encoder_arch = cfg['encoder_arch'], 
                          decoder_arch = cfg['decoder_arch'],
                          err_func     = cfg['err_func'],
                          entropyY     = data['entropyY'],
                          trainable_noisevar = cfg['train_noisevar']=='gradient', 
                          noisevar     = 0.01,
                          kdewidth    = -20.)            
            
            saver = tf.train.Saver(max_to_keep=30) # save last 30 epochs
            saver.restore(sess, tf.train.latest_checkpoint(savedir+'/basemodel'))
            sqmode  = 'sq'  if cfg['squaredIB'] else 'reg'
            print("Doing %s, beta=%0.4f, %s" % (mode, beta, sqmode))
            
            n_batch          = cfg['n_batch']
            n_noisevar_batch = cfg['n_noisevar_batch']
            saved_data       = []
            n_mini_batches   = int(len(data['trn_Y']) / n_batch)

            Ixt  = n.Ixt if mode == 'nlIB' else n.vIxt
            loss = beta*(Ixt**2 if cfg['squaredIB'] else Ixt) - n.Iyt

            trainstep  = optimizer.minimize(loss)
            sess.run(tf.variables_initializer(optimizer.variables()))

            #cdata = stats(sess, mode, beta, loss, 0, data, n, do_print=True)

            noisevar_optimizer = None

            start_time = time.time()
            for epoch in range(cfg['n_epochs']):
                # randomize order of training data
                permutation  = np.random.permutation(len(data['trn_Y']))
                train_X = data['trn_X'][permutation]
                train_Y = data['trn_Y'][permutation]

                x_batch, y_batch, dmatrix = None, None, None

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

                #cdata = stats(sess, mode, beta, loss, epoch+1, data, n, start_time=start_time, do_print=epoch % report_every == 0)
                #saved_data.append(cdata)
                #write_data(savedir, epoch+1, sess, saver, [cfg, saved_data])

#             trainutils.train(sess, saver, mode, beta, cfg, data, n, optimizer, report_every=report_every, 
#                              savedir=savedir + '/results-%s-%0.5f-%s-run%d' % (mode, beta, sqmode, runndx), 
#                              fit_var=cfg['initial_fitvar'])
            del saver

            print()
            print()




