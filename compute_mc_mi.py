import os
os.environ['THEANO_FLAGS'] = 'device=cpu'#lib.cnmem=0.19'

import numpy as np
import pyximport; pyximport.install()
from mc_entropy2 import mc_entropy2
import hashlib
import cPickle

def get_mc_mi(means, logvars, skip=50, npn=10):
    logdets = np.sum(logvars, axis=1)
    num_dims = logvars.shape[1]
    condentropy = 0.5*(num_dims*np.log((2*np.pi*np.e)) + logdets)
    means = means.astype('float32')
    logvars = logvars.astype('float32')
    H = mc_entropy2(means[::skip], logvars[::skip],num_samples_per_normal=npn)
    condH = np.mean(condentropy)
    return H - condH

nats2bit = 1.0/np.log(2)

num_skip = 10
num_per_gaussian = 10

BASE_DIR = 'models'
saved_dict_mc = {}

for fname in sorted(os.listdir(BASE_DIR)):
    if not fname.startswith('savedhist'):
        continue
    with open(BASE_DIR+'/'+fname, 'rb') as f:
        d = cPickle.load(f)
        beta = d['args']['beta']
        fsfx = fname[len('savedhist-'):-len('.dat')]
        chash = hashlib.md5(f.read()).hexdigest()

        print fsfx
        targetfname = BASE_DIR + '/mcentropy-'+fsfx+'.dat'
        if os.path.isfile(targetfname):
            with open(targetfname, 'rb') as f2:
                savedhash = cPickle.load(f2).get('hash', '')
                if savedhash != chash:
                    print targetfname, 'exists but hash doesnt match'
                    continue  # TODO REMOVE!!!
                else:
                    print targetfname, 'exists and hash matches'
                    continue
                    
        #if fsfx != 'nlIB-800-800-20-800-0.750000':
        #    continue
        
        from getactivations import get_noiselayer_activations, trn, tst
        means, logvars = get_noiselayer_activations(d['args'], fsfx, trn.X)
        mctrn = get_mc_mi(means,logvars, skip=num_skip,npn=num_per_gaussian)
        means, logvars = get_noiselayer_activations(d['args'], fsfx, tst.X)
        mctst = get_mc_mi(means,logvars, skip=num_skip,npn=num_per_gaussian)
        
        print 'Trn', mctrn, 'Tst', mctst
        
        savedict= {'trn':mctrn,
                   'tst':mctst,
                   'skip':num_skip,
                   'numsamples':num_per_gaussian,
                   'hash':chash
                  }
        with open(targetfname, 'wb') as wf:
            cPickle.dump(savedict, wf)
        