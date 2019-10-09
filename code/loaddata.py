import tensorflow as tf
import numpy as np
import scipy.io
import pickle
import entropy
import scipy.stats

def set_Y_entropy(data):
    for r in ['trn','tst']:
        Y = data[r + '_Y']
        if data['err'] == 'ce':
            v = entropy.entropy(Y.mean(axis=0))
        elif data['err'] == 'mse':
            #hist = np.histogram(Y, bins=100)
            #v = scipy.stats.rv_histogram(hist).entropy()
            if Y.shape[1] != 1: # only 1-d continuous output supported right now
                raise Exception()
            v = entropy.gaussian_entropy_np(1, np.var(Y))
        else:
            raise Exception('Unknown error func')
        data[r+'_entropyY'] = v
    return data        

def one_hot(x, n_classes=None):
    assert(np.array(x).ndim == 1)
    
    # input: 1D array of N labels, output: N x max(x)+1 array of one-hot vectors
    if n_classes is None:
        n_classes = max(x) + 1

    x_one_hot = np.zeros([len(x), n_classes])
    x_one_hot[np.arange(len(x)), x] = 1
    return x_one_hot


def load_housing():
    from sklearn.datasets import fetch_california_housing
    d=fetch_california_housing()
    d['data'] -= d['data'].mean(axis=0)
    d['data'] /= d['data'].std(axis=0)
    
    # Housing prices above 5 are all collapsed to 5, which makes the Y distribution very strange. Drop these
    d['data']   = d['data'][d['target'] < 5]
    d['target'] = d['target'][d['target'] < 5]
    
    d['target'] = np.log(d['target'])
    
    np.random.seed(12345)
    permutation = np.random.permutation(len(d['data']))
    d['data']   = d['data'][permutation]
    d['target'] = d['target'][permutation]
    
    l = int(len(d['data'])*0.8)
    
    data = {'err':'mse',
            'trn_X': d['data'][:l],
            'trn_Y': np.atleast_2d(d['target'][:l]).T,
            'tst_X': d['data'][l:],
            'tst_Y': np.atleast_2d(d['target'][l:]).T,
           }
    
    return data

def load_mnist(n_data=None, fashion_mnist=False):
    method = tf.keras.datasets.mnist if not fashion_mnist else tf.keras.datasets.fashion_mnist
    (train_data, train_labels), (test_data, test_labels) = method.load_data()

#     # randomize order
#     permutation = np.random.permutation(len(train_labels))
#     train_data = train_data[permutation]
#     train_labels = train_labels[permutation]
#     permutation = np.random.permutation(len(test_labels))
#     test_data = test_data[permutation]
#     test_labels = test_labels[permutation]

    # normalize, reshape, and convert to one-hot vectors
    train_data   = (np.reshape(train_data, (-1, 784)) / (255./2.) - 1.)
    test_data    = (np.reshape(test_data, (-1, 784)) / (255./2.) - 1.)
    train_labels = one_hot(train_labels)
    test_labels  = one_hot(test_labels)

    if n_data is not None:
        data = {'trn_X': train_data[:n_data], 'trn_Y': train_labels[:n_data], 
                'tst_X': test_data[:n_data] , 'tst_Y': test_labels[:n_data]}
    else:
        data = {'trn_X': train_data, 'trn_Y': train_labels, 
                'tst_X': test_data , 'tst_Y': test_labels}

    #data['trn_entropyY'] = np.log(10)
    #data['tst_entropyY'] = np.log(10)
    data['err'] = 'ce'
    return data

def load_delicious():
    # https://github.com/tsoumakas/mulan/tree/master/data/multi-label/delicious
    # delicious multilabel dataset
    data = {}
    for mode in ['trn','tst']:
        k = 'train' if mode == 'trn' else 'test'
        with open('data/Multilabel-Classification-Datasets-master/delicious/delicious-%s-features.pkl'%k,'rb') as f:
            data[mode+'_X'] = pickle.load(f)
        with open('data/Multilabel-Classification-Datasets-master/delicious/delicious-%s-labels.pkl'%k,'rb') as f:
            pred = pickle.load(f)
            good_ix = pred.sum(axis=1) > 0
            Y = pred[good_ix,:]
            Y = Y / Y.sum(axis=1)[:,None]
            data[mode+'_Y'] = Y

    data['err'] = 'ce'
    return data

# def load_wine():
#     mx = np.vstack([
#         np.genfromtxt('data/winequality-red.csv',delimiter=";", skip_header=1),
#         np.genfromtxt('data/winequality-white.csv',delimiter=";", skip_header=1),
#     ])
#     np.random.seed(12345)
#     permutation  = np.random.permutation(len(mx))
#     mx = mx[permutation,:]

#     X = mx[permutation,:-1]
#     y = mx[permutation,-1]
#     #Y = one_hot(y.astype('int')) # 
#     Y = np.zeros( (len(mx), 2))
#     Y[y < 6,0] = 1.0 
#     Y[y >= 6,1] = 1.0 
#     #Y[y == 5,:] = 0.5
#     ps = Y.mean(axis=0)
#     entropyY = np.sum([-p*np.log(p) for p in ps if not np.isclose(p,0)])
                      
#     hl = int(len(mx)/2)

#     data = { 'trn_X' : X[:hl,:], 'trn_Y': Y[:hl,:],
#              'tst_X' : X[hl:,:], 'tst_Y': Y[hl:,:],
#              'entropyY' : entropyY}
    
#     return data


def load_wine():
    mx = np.vstack([
        np.genfromtxt('data/winequality-red.csv',delimiter=";", skip_header=1),
        np.genfromtxt('data/winequality-white.csv',delimiter=";", skip_header=1),
    ])
    np.random.seed(12345)
    permutation  = np.random.permutation(len(mx))
    mx = mx[permutation,:]

    X = mx[:,:-1]
    y = mx[:,-1]
    
    # y ranges from 0 to 10, indicating wine quality
    # convert this to a probabilistic classification task,
    # class 1 is "bad wine" , class 2 is "good wine"
    Y = np.zeros( (X.shape[0], 2) )
    y = (y - y.mean()) / y.std()
    
    y = 1./(1.+np.exp(-10*y))
    #print(y)
    #y = y - y.min()
    #y = y / y.max()
    #y = 2*(y - 0.5)
    #print(y)
    #y = y ** 5.
    #y = (y-1.)/2
    Y[:,0] = 1. - y #  / y.max()
    Y[:,1] = y # / y.max()
    #print(Y.shape)
    #print(Y.mean(axis=0))
    
    hl = int(len(mx)/2)
    #print( entropy(Y.mean(axis=0)))
    #print(Y)
    data = { 'trn_X' : X[:hl,:], 'trn_Y': Y[:hl,:],
             'tst_X' : X[hl:,:], 'tst_Y': Y[hl:,:]}
    #data['trn_entropyY'] = entropy.entropy(data['trn_Y'].mean(axis=0))
    #data['tst_entropyY'] = entropy.entropy(data['tst_Y'].mean(axis=0))
    
    data['err'] = 'ce'
    return data




def load_szt():
    # Data from artificial dataset used in Schwartz-Ziv and Tishby
    d1 = scipy.io.loadmat('data/g1.mat')
    d2 = scipy.io.loadmat('data/g2.mat')
    data = { 'trn_X' : d1['F'].astype('float32'), 'trn_Y': one_hot(d1['y'].flat),
             'tst_X' : d2['F'].astype('float32'), 'tst_Y': one_hot(d2['y'].flat),
             'entropyY': np.log(2)}
    data['err'] = 'ce'
    return data    

def load_data(runtype, validation=False):
    if runtype == 'MNIST':
        data = load_mnist()
    elif runtype == 'FashionMNIST':
        data = load_mnist(fashion_mnist=True)
    elif runtype == 'Housing':
        data = load_housing()
    elif runtype == 'Delicious':
        data = load_delicious()
    elif runtype == 'Wine':
        data = load_wine()
    elif runtype == 'NoisyClassifier':
        data = load_szt()
    elif runtype == 'Autompg':
        with open('data/autompg.pkl', 'rb') as f:
            data = pickle.load(f)
    elif runtype == 'Regression':
        # data generated by makeregressiondata.py
        with open('data/regression-100-10.pkl', 'rb') as f:
            data = pickle.load(f)
        trn_labelcov = np.cov(data['trn_Y'].T)
        data['trn_entropyY'] = 0.5 * np.log(np.linalg.det(2*np.pi*np.exp(1)*trn_labelcov))
        tst_labelcov = np.cov(data['tst_Y'].T)
        data['tst_entropyY'] = 0.5 * np.log(np.linalg.det(2*np.pi*np.exp(1)*tst_labelcov))
        data['err'] = 'mse'
    else:
        raise Exception('unknown runtype')
    
    for k in ['trn_X','trn_Y','tst_X','tst_Y']:
        data[k] = data[k].astype(np.float32)
        
    if validation:
        cutoff = int(0.8*len(data['trn_X']))
        np.random.seed(12345)
        permutation = np.random.permutation(len(data['trn_X']))
        dX = data['trn_X'][permutation]
        dY = data['trn_Y'][permutation]
        data['trn_X'] = dX[:cutoff]
        data['trn_Y'] = dY[:cutoff]
        data['val_X'] = dX[cutoff:]
        data['val_Y'] = dY[cutoff:]
            
#     train_data = train_data[permutation]
#     train_labels = train_labels[permutation]
#     permutation = np.random.permutation(len(test_labels))
#     test_data = test_data[permutation]
#     test_labels = test_labels[permutation]
        
    data = set_Y_entropy(data)
    data['runtype'] = runtype
    
    return data
    

