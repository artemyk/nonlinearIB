# Code to estimate mixture entropy using Monte Carlo sampling

import numpy as np
import scipy
from entropy import pairwise_distance2_np

from randomgen import RandomGenerator, MT19937
rnd = RandomGenerator(MT19937())

def get_mc_entropy(mx, var):
    n, d     = mx.shape
    mx2 = mx + rnd.standard_normal(mx.shape, dtype=np.float32)*np.sqrt(var)

    dist_norm = pairwise_distance2_np(mx, mx2) / (-2.0 * var)  
    
    const     =  - 0.5  * d * np.log(2.0 * np.pi * var) - np.log(n)
    logprobs  = const + scipy.special.logsumexp(dist_norm , axis=0) 

    return -np.mean(logprobs)

