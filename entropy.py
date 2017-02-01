import numpy as np
import scipy
import keras.backend as K

def np_entropy(p):
    cp = np.log(p)
    cp[np.isclose(p,0.)]=0.
    return -p.dot(cp)

def logsumexp(mx, axis):
    cmax = K.max(mx, axis=axis)
    cmax2 = K.expand_dims(cmax, 1)
    mx2 = mx - cmax2
    return cmax + K.log(K.sum(K.exp(mx2), axis=1))

def kde_entropy(output, var):
    # Kernel density estimate of entropy, in nats

    dims = K.cast(K.shape(output)[1], K.floatx() ) 
    N    = K.cast(K.shape(output)[0], K.floatx() )
    
    normconst = (dims/2.0)*K.log(2*np.pi*var)
            
    # get dists matrix
    x2 = K.expand_dims(K.sum(K.square(output), axis=1), 1)
    dists = x2 + K.transpose(x2) - 2*K.dot(output, K.transpose(output))
    dists = dists / (2*var)
    
    lprobs = logsumexp(-dists, axis=1) - K.log(N) - normconst
    h = -K.mean(lprobs)
    
    return h

def kde_condentropy(output, var):
    # Return entropy of a multivariate Gaussian, in nats

    dims = K.cast(K.shape(output)[1], K.floatx() )
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    return normconst

def kde_entropy_from_dists_loo(dists, N, dims, var):
    # Given a distance matrix dists, return leave-one-out kernel density estimate of entropy
    # Dists should have very large values on diagonal (to make those contributions drop out)
    dists2 = dists / (2*var)
    normconst = (dims/2.0)*K.log(2*np.pi*var)
    lprobs = logsumexp(-dists2, axis=1) - np.log(N-1) - normconst
    h = -K.mean(lprobs)
    return h

