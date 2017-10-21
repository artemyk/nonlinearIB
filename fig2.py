# Example that demonstrates how to generate a figure similar to Fig.2 in Artemy Kolchinsky, Brendan D. Tracey, David H. Wolpert, "Nonlinear Information Bottleneck", https://arxiv.org/abs/1705.02436

# Minimal example of how to create a model with nonlinearIB layers
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense

import buildmodel, layers, training, reporting

BETA_VAL = 0.4

trn, tst = buildmodel.get_mnist()
input_layer = Input((trn.X.shape[1],))

# hidden_layers_to_add should include a list of all layers that will get added before the nonlinearIB layers
hidden_layers_to_add = [Dense(800, activation='relu'),
                        Dense(800, activation='relu'),
                        Dense(2, activation='linear'), ]

# *** The following creates the layers and callbacks necessary to run nonlinearIB ***
micalculator = layers.MICalculator(BETA_VAL, model_layers=hidden_layers_to_add, data=trn.X, miN=1000)
noiselayer = layers.NoiseLayer(logvar_trainable=True, test_phase_noise=False)
micalculator.set_noiselayer(noiselayer)

#    Start hooking up the layers together
cur_hidden_layer = input_layer
for l in hidden_layers_to_add:
    cur_hidden_layer = l(cur_hidden_layer)

noise_input_layer = layers.IdentityMap(activity_regularizer=micalculator)(cur_hidden_layer)
nonlinearIB_output_layer = noiselayer(noise_input_layer)

nonlinearIB_callback = training.KDETrain(mi_calculator=micalculator)
# *** Done setting up nonlinearIB stuff ***

decoder = Dense(800, activation='relu')(nonlinearIB_output_layer)

outputs = Dense(trn.nb_classes, activation='softmax')(decoder)

model = Model(inputs=input_layer, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


init_lr = 0.001
lr_decay = 0.5
lr_decaysteps = 15
import keras.callbacks
def lrscheduler(epoch):
    lr = init_lr * lr_decay**np.floor(epoch / lr_decaysteps)
    #lr = max(lr, 1e-5)
    print('Learning rate: %.7f' % lr)
    return lr
lr_callback = keras.callbacks.LearningRateScheduler(lrscheduler)

model.fit(x=trn.X, y=trn.Y, verbose=2, batch_size=128, epochs=200, 
          validation_data=(tst.X, tst.Y), 
          callbacks=[nonlinearIB_callback, lr_callback])

#%matplotlib inline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import keras.backend as K
f = K.function([model.layers[0].input, K.learning_phase()], [noiselayer.output,])
hiddenlayer_activations = f([trn.X,0])[0]

plt.figure(figsize=(5,5))
plt.scatter(hiddenlayer_activations[:,0], hiddenlayer_activations[:,1], marker='.', edgecolor='none', c=trn.y, alpha=0.05)
plt.xticks([]); plt.yticks([])

plt.savefig('fig2.png')

