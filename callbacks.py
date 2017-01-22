from keras.callbacks import Callback

 
class NoiseTrain(Callback):
    def __init__(self, traindata, noiselayer):
        super(NoiseTrain, self).__init__()
        self.traindata = traindata
        self.noiselayer = noiselayer
        
    def on_train_begin(self, logs={}):
        modelobj = self.model.model
        inputs = modelobj.inputs + modelobj.targets + modelobj.sample_weights + [ K.learning_phase(),]
        lossfunc = K.function(inputs, [modelobj.total_loss])
        jacfunc  = K.function(inputs, K.gradients(modelobj.total_loss, self.noiselayer.logvar))
        sampleweights = np.ones(len(self.traindata.X))
        def obj(logvar):
            v = K.get_value(self.noiselayer.logvar)
            K.set_value(self.noiselayer.logvar, logvar.flat[0])
            r = lossfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])[0]
            K.set_value(self.noiselayer.logvar, v)
            return r
        def jac(logvar):
            v = K.get_value(self.noiselayer.logvar)
            K.set_value(self.noiselayer.logvar, logvar.flat[0])
            r = np.atleast_2d(np.array(jacfunc([self.traindata.X, self.traindata.Y, sampleweights, 1])))[0]
            K.set_value(self.noiselayer.logvar, v)
            return r
            
        self.obj = obj # lambda logvar: lossfunc([self.traindata.X_train, self.traindata.Y_train, self.sampleweights, logvar[0], 1])[0]
        self.jac = jac # lambda logvar: np.array(jacfunc([self.traindata.X_train, self.traindata.Y_train, self.sampleweights, logvar[0], 1]))
    
    def on_epoch_begin(self, epoch, logs={}):
        r = scipy.optimize.minimize(self.obj, K.get_value(self.noiselayer.logvar), jac=self.jac)
        best_val = r.x[0]
        cval =  K.get_value(self.noiselayer.logvar)
        max_var = 1.0 + cval
        if best_val > max_var:
            # don't raise it too fast, so that gradient information is preserved 
            best_val = max_var
            
        K.set_value(self.noiselayer.logvar, best_val)
        #print 'noiseLV=%.5f' % K.get_value(self.noiselayer.logvar)
        
class KDETrain(Callback):
    def __init__(self, entropy_train_data, kdelayer, *kargs, **kwargs):
        super(KDETrain, self).__init__(*kargs, **kwargs)
        self.kdelayer = kdelayer
        self.entropy_train_data = entropy_train_data
        
    def on_train_begin(self, logs={}):
        self.nlayerinput = lambda x: K.function([self.model.layers[0].input], [self.kdelayer.input])([x])[0]
        N, dims = self.entropy_train_data.shape
        Kdists = K.placeholder(ndim=2)
        Klogvar = K.placeholder(ndim=0)
        def obj(logvar, dists):
            #print 'here', logvar # lossfunc([dists, logvar[0]])[0]
            return lossfunc([dists, logvar.flat[0]])[0]
        def jac(logvar, dists):
            #print logvar, lossfunc([dists, logvar[0]]), jacfunc([dists, logvar[0]])
            return np.atleast_2d(np.array(jacfunc([dists, logvar.flat[0]])))[0] 
            
        lossfunc = K.function([Kdists, Klogvar,], [kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar))])
        jacfunc  = K.function([Kdists, Klogvar,], K.gradients(kde_entropy_from_dists_loo(Kdists, N, dims, K.exp(Klogvar)), Klogvar))
        self.obj =obj #  lambda logvar, dists: np.array([lossfunc([dists, logvar[0]]),]) # [0]
        self.jac =jac # lambda logvar, dists: jacfunc([dists, np.array([logvar]).flat[0]])[0]

    @staticmethod
    def get_dists(output):
        N, dims = output.shape

        # Kernel density estimation of entropy
        y1 = output[None,:,:]
        y2 = output[:,None,:]

        dists = np.sum((y1-y2)**2, axis=2) 
        return dists
    
    def on_epoch_begin(self, epoch, logs={}):
        vals = self.nlayerinput(self.entropy_train_data)
        dists = self.get_dists(vals)
        dists += 10e20 * np.eye(dists.shape[0])
        r = scipy.optimize.minimize(self.obj, K.get_value(self.kdelayer.logvar).flat[0], 
                                    jac=self.jac, 
                                    args=(dists,),
                                    )
        best_val = r.x.flat[0]
        K.set_value(self.kdelayer.logvar, best_val)
        #print 'kdeLV=%.5f' % K.get_value(self.kdelayer.logvar)


class ReportVars(Callback):
    def __init__(self, kdelayer, noiselayer, *kargs, **kwargs):
        super(ReportVars, self).__init__(*kargs, **kwargs)
        self.noiselayer = noiselayer
        self.kdelayer = kdelayer
        
    def on_epoch_end(self, epoch, logs={}):
        lv1 = K.get_value(self.kdelayer.logvar)
        lv2 = K.get_value(self.noiselayer.logvar)
        logs['kdeLV']   = lv1
        logs['noiseLV'] = lv2
        print 'kdeLV=%.5f, noiseLV=%.5f' % (lv1, lv2)  

