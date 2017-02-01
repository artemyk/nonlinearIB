cimport cython
cimport numpy as np
import  numpy as np

@cython.boundscheck(False)  # these don't contribute much in this example
@cython.wraparound(False)   # but are often useful for numeric arrays


def mc_entropy2(np.ndarray[np.float32_t, ndim=2] means, 
                np.ndarray[np.float32_t, ndim=2] logvars, 
                int num_samples_per_normal=1):
    cdef int num_inputs = means.shape[0]
    cdef int num_dims   = means.shape[1]
    
    cdef np.ndarray[np.float32_t, ndim=1] logdets = np.sum(logvars, axis=1)
    cdef np.ndarray[np.float32_t, ndim=1] lnormalizations = 0.5*(num_dims*np.log(2.*np.pi) + logdets)
    
    cdef np.ndarray[np.float32_t, ndim=2] sigmas = np.exp(0.5*logvars)
    #x = x * sigmas[:,:,None] + means[:,:,None]
    cdef np.ndarray[np.float32_t, ndim=3] x = \
       np.random.randn(num_inputs, num_dims, num_samples_per_normal).astype('float32')*sigmas[:,:,None] + means[:,:,None]
    
    cdef np.ndarray[np.float32_t, ndim=2] inv_sigmas = 1./sigmas
    
    cdef np.ndarray[np.float32_t, ndim=2] probs = np.zeros([num_inputs, num_samples_per_normal], dtype=np.float32)
    cdef int i ,j
    cdef np.ndarray[np.float32_t, ndim=2] zs = np.zeros([num_inputs, num_samples_per_normal], dtype=np.float32)
    for i in range(num_inputs):
        zs = zs*0.0
        for j in range(num_dims):
            zs += ((x[:,j,:]-means[i,j])*inv_sigmas[i,j])**2
        probs += np.exp( -0.5*zs-lnormalizations[i])
    probs /= len(means)
    entropy = -np.mean(np.log(probs))
    return entropy


"""
def mc_entropy2(float[:,:] means, 
                float[:,:] logvars, 
                int num_samples_per_normal=1):
    cdef int num_inputs = means.shape[0]
    cdef int num_dims   = means.shape[1]
    
    cdef float[:] logdets = np.sum(logvars, axis=1)
    cdef float[:] lnormalizations = 0.5*(num_dims*np.log(2.*np.pi) + logdets)
    
    cdef float[:,:] sigmas = logvars
    cdef float[:,:] inv_sigmas = logvars
    cdef int i ,j
    for i in range(num_inputs):
        for j in range(num_dims):
            sigmas[i,j] = exp(0.5*logvars[i,j])
            inv_sigmas[i,j] = 1./sigmas[i,j]
    
    #cdef i = np.exp(float(0.5)*logvars)
    #x = x * sigmas[:,:,None] + means[:,:,None]
    cdef float[:,:,:] x = \
       np.random.randn(num_inputs, num_dims, num_samples_per_normal).astype('float32')*sigmas[:,:,None] + means[:,:,None]
    
    #cdef float[:,:] inv_sigmas = 1./sigmas
    
    cdef float[:,:] probs = np.zeros([num_inputs, num_samples_per_normal], dtype=np.float32)
    cdef float[:,:] zs = np.zeros([num_inputs, num_samples_per_normal], dtype=np.float32)

    #cdef float[:,:] zs 
    with nogil, parallel():
        for i in prange(num_inputs, nogil=True):
            #zs = zs*0.0
            #zs = np.ascontiguousarray(zs0, dtype=ctypes.c_float32)

            for j in range(num_dims):
                zs = zs + ((x[:,j,:]-means[i,j])*inv_sigmas[i,j])**2
            probs += exp( -0.5*zs-lnormalizations[i])
            
    probs /= len(means)
    entropy = -np.mean(np.log(probs))
    return entropy

"""