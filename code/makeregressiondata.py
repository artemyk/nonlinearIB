import scipy.stats
import numpy as np
import pickle

inputdims  = 100
outputdims = 1

n_trn_samples = 5000
n_tst_samples = 5000

dims = inputdims + outputdims
cov = scipy.stats.wishart.rvs(df=dims, scale=.1*np.eye(dims)+1.)
mx = np.random.multivariate_normal(mean=np.zeros(dims), cov=cov, size=(n_trn_samples+n_tst_samples,))

data = { 'trn_X': mx[:n_trn_samples,:inputdims], 'trn_Y': mx[:n_trn_samples,inputdims:],
         'tst_X': mx[n_trn_samples:,:inputdims], 'tst_Y': mx[n_trn_samples:,inputdims:]}
fname = 'data/regression-%d-%d.pkl'%(inputdims, outputdims)
with open(fname, 'wb') as f:
    pickle.dump(data, f)

print("Saved in", fname)
