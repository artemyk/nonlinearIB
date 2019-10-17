# This is a simple demo showing how to apply Nonlinear IB (https://arxiv.org/abs/1705.02436)
# to the MNIST dataset.  Importantly, we use two stage training: we first train the network
# to have good predition (no compression term), we then train while penalizing compression

from __future__ import print_function

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth=True

import utils

# Configuration settings
squaredIB        = True         # Whether to minimize beta*I(X;T) - I(Y;T) or beta*I(X;T)^2 - I(Y;T)
batchsize        = 256          # Mini batch size
report_every     = 10           # How often to report
patience         = 10           # Early stopping patience -- # epochs to go without improvement on validation data
beta             = 0.05         # IB trade-off parameter


# Load data
import loaddata
data           = loaddata.load_data('MNIST', validation=True)
input_dim      = data['trn_X'].shape[1]
output_dim     = data['trn_Y'].shape[1]


# Build the network
tf.reset_default_graph()

import iblayer
iblayerobj     = iblayer.NoisyIBLayer()

layers = []
layers.append( tf.placeholder(tf.float32, [None,input_dim,], name='X' ) )
layers.append( tf.keras.layers.Dense(128, activation=tf.nn.relu)(layers[-1]) )
layers.append( tf.keras.layers.Dense(128, activation=tf.nn.relu)(layers[-1]) )
layers.append( tf.keras.layers.Dense(10 , activation=None)(layers[-1]) )
layers.append( iblayerobj(layers[-1]) )
layers.append( tf.keras.layers.Dense(128, activation=tf.nn.relu)(layers[-1]) )
layers.append( tf.keras.layers.Dense(output_dim, activation=None)(layers[-1]) )

inputs         = layers[0]
predictions    = layers[-1]
true_outputs   = tf.placeholder(tf.float32, [None,output_dim,], name='trueY')

# Define statistics and optimization terms
f              = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_outputs, logits=predictions)
cross_entropy  = utils.get_error   ('ce', y_pred=predictions, y_true=true_outputs)
accuracy       = utils.get_accuracy('ce', y_pred=predictions, y_true=true_outputs)

Ixt            = iblayerobj.Ixt
entropyY       = tf.placeholder(dtype=tf.float32, shape=(), name='entropyY')
Iyt            = entropyY - cross_entropy

optimizer      = tf.train.AdamOptimizer() 

ib_objective = Iyt - beta*Ixt**(2 if squaredIB else 1)
trainstep = optimizer.minimize(-ib_objective)
              
stats = {'Iyt':Iyt, 'Ixt':Ixt, 'acc':accuracy, 'ce':cross_entropy, 'IB':ib_objective}
    
    
# Run the training
with tf.Session(config=tfconfig) as sess:
    sess.run(tf.global_variables_initializer())

    last_val_increase_epoch = 0
    last_best_val           = None
            
    for epoch in range(300):
        
        vals =  {'epoch' : epoch}
        for mode in ['trn','val','tst']:
            batch  = np.random.choice(len(data[mode+'_Y']), 2000)
            X = data[mode+'_X'][batch]
            Y = data[mode+'_Y'][batch]
            d = sess.run(stats, feed_dict={inputs: X, true_outputs: Y, entropyY: data[mode+'_entropyY']})
            for k, v in d.items():
                vals[mode + '_' + k] = v
                    
        if epoch % report_every == 0:
            print(("Epoch {epoch:3d} : I(Y:T)={trn_Iyt:5.3f}/{val_Iyt:5.3f}/{tst_Iyt:5.3f} " + 
                                      "I(X:T)={trn_Ixt:5.3f}/{val_Ixt:5.3f}/{tst_Ixt:5.3f} " +
                                         "Acc={trn_acc:0.3f}/{val_acc:0.3f}/{tst_acc:0.3f} " +
                                          "CE={trn_ce:6.3f}/{val_ce:6.3f}/{tst_ce:6.3f} " +
                                          "IB={trn_IB:5.3f}/{val_IB:5.3f}/{tst_IB:5.3f} ").format(**vals))
            
        # Early stopping checks
        if last_best_val is None or vals['val_IB'] >= last_best_val:
            last_val_increase_epoch = epoch
            last_best_val = vals['val_IB']

        if (epoch - last_val_increase_epoch) >= patience:
            print("Validation hasn't improved since epoch %d -- quiting" % last_val_increase_epoch)
            break
        
        for batch in utils.get_train_batches(data['trn_X'], data['trn_Y'], batchsize):
            batch['entropyY:0'] = data['trn_entropyY'] 
            sess.run(trainstep, feed_dict=batch)
            
            
            
