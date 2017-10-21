import keras
import nonlinearib, utils

trn, tst = utils.get_mnist()

input_layer    = keras.layers.Input((trn.X.shape[1],))
hidden_output  = keras.layers.Dense(10, activation='relu')(input_layer)

nlIB_layer     = nonlinearib.NonlinearIB(beta=0.01)
nlIB_output    = nlIB_layer(hidden_output)

outputs        = keras.layers.Dense(trn.nb_classes, activation='softmax')(nlIB_output)
model          = keras.models.Model(inputs=input_layer, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

nlIB_callbacks = nlIB_layer.get_training_callbacks(model, trn=trn, minibatchsize=1000)

r = model.fit(x=trn.X, y=trn.Y, validation_data=(tst.X, tst.Y), 
              batch_size=64, epochs=100, 
              callbacks=nlIB_callbacks,
              verbose=2)
