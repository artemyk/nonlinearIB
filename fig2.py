# Example that demonstrates how to generate a figure similar to Fig.2 in 
# A Kolchinsky, BD Tracey, DH Wolpert, "Nonlinear Information Bottleneck", https://arxiv.org/abs/1705.02436
from __future__ import print_function
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import keras.backend as K
import keras

import nonlinearib, reporting, utils

trn, tst = utils.get_mnist()

input_layer    = Input((trn.X.shape[1],))
hidden_output  = Dense(800, activation='relu')(input_layer)
hidden_output  = Dense(800, activation='relu')(hidden_output)
hidden_output  = Dense(2 , activation='linear')(hidden_output)

nlIB_layer     = nonlinearib.NonlinearIB(beta=0.4, noise_logvar_train_firstepoch=10)
nlIB_output    = nlIB_layer(hidden_output)

decoder_output = Dense(800, activation='relu')(nlIB_output)

outputs        = Dense(trn.nb_classes, activation='softmax')(decoder_output)

model          = Model(inputs=input_layer, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

init_lr, lr_decay, lr_decaysteps = 0.001, 0.5, 15
def lrscheduler(epoch):
    lr = init_lr * lr_decay**np.floor(epoch / lr_decaysteps)
    print('Learning rate: %.7f' % lr)
    return lr
lr_callback = keras.callbacks.LearningRateScheduler(lrscheduler)


callbacks = nlIB_layer.get_training_callbacks(model, trn=trn, minibatchsize=1000)
callbacks.append(lr_callback)
#callbacks.append(reporting.Reporter(trn=trn, tst=tst, verbose=2))

r = model.fit(x=trn.X, y=trn.Y, verbose=2, batch_size=64, epochs=100, validation_data=(tst.X, tst.Y), callbacks=callbacks)


f = K.function([model.layers[0].input, K.learning_phase()], [nlIB_layer.output,])
hiddenlayer_activations = f([trn.X,0])[0]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Plot results
plt.figure(figsize=(5,5))
plt.scatter(hiddenlayer_activations[:,0], hiddenlayer_activations[:,1], marker='.', edgecolor='none', c=trn.y, alpha=0.05)
plt.xticks([]); plt.yticks([])

plt.savefig('fig2_v3.png')

