import keras.backend as K
from keras.callbacks import Callback
import scipy.optimize
from entropy import *


# class NoiseTrain(Callback):
#     def __init__(self, traindata, noiselayer):
#         super(NoiseTrain, self).__init__()
#         self.traindata = traindata
#         self.noiselayer = noiselayer
        
#     def on_train_begin(self, logs={}):
#         modelobj = self.model.model
#         inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
#         lossfunc = K.function(inputs, [modelobj.total_loss])
#         jacfunc  = K.function(inputs, K.gradients(modelobj.total_loss, self.noiselayer.logvar))
#         sampleweights = np.ones(len(self.traindata.X))
#         def obj(logvar):
#             v = K.get_value(self.noiselayer.logvar)
#             K.set_value(self.noiselayer.logvar, logvar.flat[0])
#             r = lossfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])[0]
#             K.set_value(self.noiselayer.logvar, v)
#             return r
#         def jac(logvar):
#             v = K.get_value(self.noiselayer.logvar)
#             K.set_value(self.noiselayer.logvar, logvar.flat[0])
#             r = np.atleast_2d(np.array(jacfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])))[0]
#             K.set_value(self.noiselayer.logvar, v)
#             return r
            
#         self.obj = obj
#         self.jac = jac
        
#     def on_epoch_begin(self, epoch, logs={}):
#         r = scipy.optimize.minimize(self.obj, K.get_value(self.noiselayer.logvar), jac=self.jac)
#         best_val = r.x[0]
#         cval =  K.get_value(self.noiselayer.logvar)
#         max_var = 1.0 + cval
#         if best_val > max_var:
#             # don't raise it too fast, so that gradient information is preserved 
#             best_val = max_var
            
#         K.set_value(self.noiselayer.logvar, best_val)

        

class KDETrain(Callback):
    def __init__(self, mi_calculator, *kargs, **kwargs):
        super(KDETrain, self).__init__(*kargs, **kwargs)
        self.mi_calculator = mi_calculator
        
    def on_train_begin(self, logs={}):
        N    = self.mi_calculator.miN
        dims = self.mi_calculator.data.shape[1]
        Kdists = K.placeholder(ndim=2)
        Klogvar = K.placeholder(ndim=0)
            
        lossfunc = K.function([Kdists, Klogvar,], [kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar))])
        jacfunc  = K.function([Kdists, Klogvar,], K.gradients(kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar)), Klogvar))

        def obj(logvar, dists):
            return lossfunc([dists, logvar.flat[0]])[0]
        def jac(logvar, dists):
            return np.atleast_2d(np.array(jacfunc([dists, logvar.flat[0]])))[0] 

        self.obj = obj
        self.jac = jac

    @staticmethod
    def get_dists(output):
        N, dims = output.shape

        # Kernel density estimation of entropy
        y1 = output[None,:,:]
        y2 = output[:,None,:]

        dists = np.sum((y1-y2)**2, axis=2) 
        return dists
    
    def on_epoch_begin(self, epoch, logs={}):
        vals = K.eval(self.mi_calculator.sample_noise_layer_input)
        dists = self.get_dists(vals)
        dists += 10e20 * np.eye(dists.shape[0])
        r = scipy.optimize.minimize(self.obj, K.get_value(self.mi_calculator.kde_logvar).flat[0], 
                                    jac=self.jac, 
                                    args=(dists,),
                                    )
        best_val = r.x.flat[0]
        K.set_value(self.mi_calculator.kde_logvar, best_val)

 