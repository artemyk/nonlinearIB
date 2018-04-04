# Implementation of Deep Variational Information Bottleneck, Alemi et al.
from __future__ import print_function

import keras.backend as K
import numpy as np
from keras.layers import Layer
from keras import regularizers



class NoiseLayerVIB(Layer):
    def __init__(self, 
                 mean_dims,
                 test_phase_noise=False,
                 *kargs, **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        
        self.test_phase_noise = test_phase_noise
        self.mean_dims = mean_dims
        
        super(NoiseLayerVIB, self).__init__(*kargs, **kwargs)
        
    def get_noise(self, sigmas):
        return sigmas * K.random_normal(shape=K.shape(sigmas), mean=0., stddev=1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.mean_dims)
    
    def get_means_sigmas(self, x):
        means, rawsigmas = x[:,:self.mean_dims], x[:,self.mean_dims:]
        sigmas = K.log(1+K.exp(rawsigmas - 5.))
        return means, sigmas
    
    def call(self, x, mask=None):
        means, sigmas = self.get_means_sigmas(x)
        with_noise = means + self.get_noise(sigmas)
        if self.test_phase_noise:
            return with_noise
        else:
            return K.in_train_phase(with_noise, means) 

class MICalculatorVIB(regularizers.Regularizer):
    def __init__(self, beta):
        self.beta = beta
        super(MICalculatorVIB, self).__init__()
        
    def set_noiselayer(self, noiselayer):
        self.noiselayer = noiselayer
        
    def get_mi(self, x):
        # Compute average KL: 
        #   KL(N(u,Cov) || N(0,1)) = 0.5 * [tr(Cov) + ||u||^2 - k - ln ( |Cov| )]
        k = self.noiselayer.mean_dims
        means, sigmas = self.noiselayer.get_means_sigmas(x)
        cvars = K.square(sigmas)
        norms = K.square(means)
        norms = K.sum(norms, axis=1)
        v = 0.5*(K.sum(cvars, axis=1) + norms - float(k) - K.sum(K.log(cvars), axis=1))
        kl = K.mean(v)
        return kl
        
    def __call__(self, x):
        mi = self.get_mi(x)
        # If uncommented, returns sum, rather than average, KL across minibatch
        # mi = K.cast(K.shape(x)[0], K.floatx() ) * mi
        return K.in_train_phase(self.beta * mi, K.variable(0.0))
    

if __name__ == "__main__":
    from keras.models import Model
    from keras.layers import Input, Dense
    import keras.callbacks
    import logging
    logging.getLogger('keras').setLevel(logging.INFO)

    import argparse
    parser = argparse.ArgumentParser(description='Run Variational IB', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('beta' , type=float, default=0.0, help='beta hyperparameter value')
    parser.add_argument('--epoch_report_mi', action='store_true', default=False, help='Report MI values every epoch?')
    args = parser.parse_args()
    print("Running variational IB with beta=%.5f" % args.beta)
    
    from buildmodel import get_mnist
    trn, tst = get_mnist()

    # Build model
    micalculator = MICalculatorVIB(args.beta)
    noiselayer = NoiseLayerVIB(mean_dims=256, test_phase_noise=True)
    micalculator.set_noiselayer(noiselayer)
    
    inputs = Input(shape=(784,))
    c_layer = Dense(1024, kernel_initializer='he_uniform', activation='relu')(inputs)
    c_layer = Dense(1024, kernel_initializer='he_uniform', activation='relu')(c_layer)
    c_layer = Dense(512 , kernel_initializer='he_uniform', activation='linear', activity_regularizer=micalculator)(c_layer)
    
    c_layer = noiselayer(c_layer)
    predictions = Dense(trn.nb_classes, kernel_initializer='he_uniform', activation='softmax')(c_layer)
    model = Model(inputs=inputs, outputs=predictions)
    
    # Set up callbacks
    cbs = []
 
    def lrscheduler(epoch):
        lr = 0.0001 * 0.97**np.floor(epoch / 2)
        print('Learning rate: %.7f' % lr)
        return lr
    cbs.append(keras.callbacks.LearningRateScheduler(lrscheduler))
    
    import reporting
    reporter = reporting.Reporter(trn=trn, tst=tst, noiselayer=noiselayer, micalculator=micalculator,
                                 on_epoch_report_mi=args.epoch_report_mi)
    cbs.append(reporter)

    import keras.optimizers
    adam = keras.optimizers.Adam(lr=0.0001, beta_1 = 0.5, beta_2=0.999) # exponential weight decay?
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    try:
        model.fit(x=trn.X, y=trn.Y, verbose=2, batch_size=100, epochs=200, validation_data=(tst.X, tst.Y), callbacks=cbs)
    except KeyboardInterrupt:
        print("KeyboardInterrupt called")
    

    # Print and save results
    probs = 0.
    NUM_SAMPLES = 12
    for _ in range(NUM_SAMPLES):
        probs += model.predict(tst.X)
    probs /= float(NUM_SAMPLES)
    preds = probs.argmax(axis=-1)
    print('Final accuracy (using %d samples): %0.5f' % (NUM_SAMPLES, np.mean(preds == tst.y)))

    logs = reporter.get_logs(calculate_mi=True, calculate_loss=True)
    print('# ENDRESULTS: ',sep="")
    for k, v in logs.items():
        print("%s=%s "%(k,v), sep="")
    print()

