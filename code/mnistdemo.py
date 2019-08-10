# This is a simple demo showing how to apply Nonlinear IB (https://arxiv.org/abs/1705.02436)
# to the MNIST dataset.  Importantly, we use two stage training: we first train the network
# to have good predition (no compression term), we then train while penalizing compression

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tfconfig = tf.ConfigProto() ; tfconfig.gpu_options.allow_growth=True

# Configuration settings
squaredIB        = True         # Whether to minimize beta*I(X;T) - I(Y;T) or beta*I(X;T)^2 - I(Y;T)
n_batch          = 128          # Mini batch size
n_noisevar_batch = 1000         # Batch size for estimating width of KDE estimator (eta parameter)
report_every     = 10           # How often to report

beta             = 0.1          # IB trade-off parameter


# Load data
import loaddata
data           = loaddata.load_data('MNIST')
input_dim      = data['trn_X'].shape[1]
output_dim     = data['trn_Y'].shape[1]


# Build the network
tf.reset_default_graph()

import iblayer
iblayerobj     = iblayer.NoisyIBLayer(n_noisevar_batch=n_noisevar_batch)
layers = []
layers.append( tf.placeholder(tf.float32, [None,input_dim,] ) )
layers.append( tf.keras.layers.Dense(512, activation=tf.nn.relu)(layers[-1]) )
layers.append( tf.keras.layers.Dense(512, activation=tf.nn.relu)(layers[-1]) )
layers.append( tf.keras.layers.Dense(2  , activation=None)(layers[-1]) )
layers.append( iblayerobj(layers[-1]) )
layers.append( tf.keras.layers.Dense(512, activation=tf.nn.relu)(layers[-1]) )
layers.append( tf.keras.layers.Dense(output_dim, activation=None)(layers[-1]) )

inputs         = layers[0]
predictions    = layers[-1]
true_outputs   = tf.placeholder(tf.float32, [None,output_dim,])

# Define statistics and optimization terms
f              = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_outputs, logits=predictions)
cross_entropy  = tf.reduce_mean(f)
accuracy       = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(predictions, 1), tf.argmax(true_outputs, 1)), tf.float32))
Ixt            = iblayerobj.Ixt
Iyt            = data['entropyY'] - cross_entropy

optimizer      = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999) 

# We use two different training regimes:
# In the first, we minimize cross-entropy only, and do not change phi (noise variance)
var_list = [v for v in tf.trainable_variables() if v not in [iblayerobj.phi,]]
trainstep_ce = optimizer.minimize(cross_entropy, var_list=var_list)

# In the second, we minimize IB functional, w.r.t. 
if squaredIB:
    ib_loss = beta*Ixt**2 - Iyt
else:
    ib_loss = beta*Ixt - Iyt
trainstep_ib = optimizer.minimize(ib_loss)
              
    
    
# Run the training
with tf.Session(config=tfconfig) as sess:
    sess.run(tf.global_variables_initializer())

    cur_trainstep = trainstep_ce # Start by only minimizing cross entropy
    
    for epoch in range(200):
        # randomize order of training data
        permutation  = np.random.permutation(len(data['trn_Y']))
        train_X = data['trn_X'][permutation]
        train_Y = data['trn_Y'][permutation]
              
        if epoch % report_every == 0:
            tst_batch  = np.random.choice(len(data['tst_Y']), 2000)
            test_X = data['tst_X'][tst_batch]
            test_Y = data['tst_Y'][tst_batch]
              
            stats = [Iyt, Ixt, accuracy, cross_entropy, ib_loss]
            trn_vals = sess.run(stats, feed_dict={inputs: train_X[:2000], true_outputs: train_Y[:2000]})
            tst_vals = sess.run(stats, feed_dict={inputs: test_X        , true_outputs: test_Y})
            print("Epoch %4d : I(Y:T)=%6.3f/%6.3f I(X:T)=%6.3f/%6.3f Accuracy=%0.3f/%0.3f CE=%6.3f/%6.3f IB=%6.3f/%6.3f" % 
                  (epoch, trn_vals[0], tst_vals[0], trn_vals[1], tst_vals[1], trn_vals[2], tst_vals[2], 
                   trn_vals[3], tst_vals[3], trn_vals[4], tst_vals[4],) )

        # TODO: af
        iblayerobj.optimize_eta(sess, inputs, train_X, true_outputs, train_Y)

        if epoch == 100:
            print("** Switching to IB optimization **")
            cur_trainstep = trainstep_ib

        for batch in range(int(len(data['trn_Y']) / n_batch)):
            x_batch = train_X[batch * n_batch:(1 + batch) * n_batch]
            y_batch = train_Y[batch * n_batch:(1 + batch) * n_batch]
            cparams = {inputs: x_batch, true_outputs: y_batch}
              
            sess.run(cur_trainstep, feed_dict=cparams)
