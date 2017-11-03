# Minimal example of how to create a model with nonlinearIB layers

import os ; os.environ['KERAS_BACKEND']='tensorflow'

from keras.models import Model
from keras.layers import Input, Dense

import buildmodel, layers, training, reporting

trn, tst = buildmodel.get_mnist()
BETA_VAL = 1e-1

input_layer      = Input((trn.X.shape[1],))

# *** The following creates the layers and callbacks necessary to run nonlinearIB ***

# hidden_layers_to_add should include a list of all layers that will get added before the nonlinearIB layers
hidden_layers_to_add = [ Dense(20, activation='relu'), ] # Hidden layer with 20 hidden units

micalculator = layers.MICalculator(BETA_VAL, hidden_layers_to_add, data=trn.X)
noiselayer = layers.NoiseLayer()
micalculator.set_noiselayer(noiselayer)

# Start hooking up the layers together
cur_hidden_layer = input_layer
for l in hidden_layers_to_add:
    cur_hidden_layer = l(cur_hidden_layer)

noise_input_layer = layers.IdentityMap(activity_regularizer=micalculator)(cur_hidden_layer)
nonlinearIB_output_layer = noiselayer(noise_input_layer)

nonlinearIB_callback = training.KDETrain(mi_calculator=micalculator)
# *** Done setting up nonlinearIB stuff ***

outputs = Dense(trn.nb_classes, activation='softmax')(nonlinearIB_output_layer)

model = Model(inputs=input_layer, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=trn.X, y=trn.Y, verbose=2, batch_size=128, epochs=60, 
          validation_data=(tst.X, tst.Y), callbacks=[nonlinearIB_callback])

