"""
Check the correctness of gor on HardNet loss using multiple GPUs
Usage: check_gor_HardNet.py

Author: Xu Zhang
Email: xu.zhang@columbia.edu.cn
"""

#! /usr/bin/env python2

import numpy as np
import scipy.io as sio
import time
import os
import sys
import pandas as pd
import subprocess
import shlex
import argparse
####################################################################
# Parse command line
####################################################################
def usage():
    print >> sys.stderr 
    sys.exit(1)

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

gpu_set = ['0']
#parameter_set = ['nlIB']
parameter_set = ['0.4']

number_gpu = len(gpu_set)

#datasets = ['notredame', 'yosemite', 'liberty']
process_set = []


for idx, parameter in enumerate(parameter_set):
    print('Test Parameter: {}'.format(parameter))
    command = 'python run.py --nb_epoch 60 --mode=nlIB --beta={} --gpu-id {} --backend tensorflow'\
            .format(parameter, gpu_set[idx%number_gpu])

    print(command)
    p = subprocess.Popen(shlex.split(command))
    process_set.append(p)
    
    if (idx+1)%number_gpu == 0:
        print('Wait for process end')
        for sub_process in process_set:
            sub_process.wait()
    
        process_set = []

    time.sleep(60)

for sub_process in process_set:
    sub_process.wait()

