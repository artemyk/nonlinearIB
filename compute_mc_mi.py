import os
os.environ['THEANO_FLAGS'] = 'device=cpu'

import numpy as np
import pyximport; pyximport.install()
from mc_entropy2 import mc_entropy2
import hashlib
import cPickle
import argparse

parser = argparse.ArgumentParser(description='Compute MC entropy',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('path', nargs='*', help='Path of a file or a folder of files.')
args = parser.parse_args()

def get_mc_mi(means, logvars, skip=50, npn=10):
    logdets = np.sum(logvars, axis=1)
    num_dims = logvars.shape[1]
    condentropy = 0.5*(num_dims*np.log((2*np.pi*np.e)) + logdets)
    means = means.astype('float32')
    logvars = logvars.astype('float32')
    H = mc_entropy2(means[::skip], logvars[::skip],num_samples_per_normal=npn)
    condH = np.mean(condentropy)
    return H - condH

def get_hash(pathname):
    with open(pathname, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()
    
nats2bit = 1.0/np.log(2)

num_skip = 10
num_per_gaussian = 20

BASE_DIR = 'models'
saved_dict_mc = {}


if args.path is not None and len(args.path):
    print "Doing file(s)", args.path
    files = args.path
else:
    files = [BASE_DIR+'/'+f for f in sorted(os.listdir(BASE_DIR))]

for pathname in files:
    cbase, fname = os.path.split(pathname)
    if not fname.startswith('savedhist'):
        continue
    with open(pathname, 'rb') as f:
        d = cPickle.load(f)
        beta = d['args']['beta']
        fsfx = fname[len('savedhist-'):-len('.dat')]
        chash = get_hash(cbase + '/fitmodel-'+fsfx+'.h5')

        print fsfx
        targetfname = cbase + '/mcentropy-'+fsfx+'.dat'
        if os.path.isfile(targetfname):
            with open(targetfname, 'rb') as f2:
                saveddict = cPickle.load(f2)
                if saveddict.get('hash', '') != chash:
                    print targetfname, 'exists but hash doesnt match'
                elif saveddict['numsamples'] != num_per_gaussian or saveddict['skip'] != num_skip:
                    print targetfname, 'exists but numsamples (%d, %d) or num_skip (%d, %d) doesnt match' % (num_per_gaussian, saveddict['numsamples'], num_skip, saveddict['skip']) 
                else:
                    print targetfname, 'exists and hash/numsamples/skip matches'
                    continue
                    
        #if fsfx != 'nlIB-800-800-20-800-0.750000':
        #    continue
        
        from getactivations import get_noiselayer_activations, trn, tst
        means, logvars = get_noiselayer_activations(d['args'], fsfx, trn.X, basedir=cbase)
        dims = means.shape[1]
        if dims > 150:
            print "Too many dimensions (%d), skipping" % dims
            continue
        mctrn = get_mc_mi(means,logvars, skip=num_skip,npn=num_per_gaussian)
        means, logvars = get_noiselayer_activations(d['args'], fsfx, tst.X, basedir=cbase)
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
        
