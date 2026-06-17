#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 16:14:39 2025

@author: veroneze
"""

import socket
import os
import sys
import subprocess
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils_PM import getChi2CriticalValue, chi2Tominsups

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


### Set these values:

# Choose the probability of exceeding the crtitical value of chi2:
alpha = 0.05

# Miminum relative minimum support:
rmminsup = 0.01

# Miminum absolute minimum support:
mminsup = 2

# Minimum confidence
minconf = 0.51

# Folder to save the data files for RIn-Close:
FOUTPUT = '/Documents/AIDE_rinclose/'
FBASE_CECI = '/CECI/home/ucl/ingi/rveronez'
###############################################################################


id_uci = int(sys.argv[1])
nsites = int(sys.argv[2])
seed = int(sys.argv[3])
print('Dataset:', id_uci, 'n_sites:', nsites, 'seed:', seed)

# Ids UCI that requires a special treatment
ids_uci_special = [22, 26, 69]


if socket.gethostname() == 'PC-SE24-099':
    fpath = os.path.expanduser('~'+FOUTPUT)
else:
    fpath = FBASE_CECI + FOUTPUT
fpath_s = fpath + f'nsites{nsites}/'
frinclose = f'{fpath}RInClose_mminsup'


def get_minsups(id_uci, counts, theta):
    # Get the minsup for each class label based on the chi2
    rminsups, minsups = chi2Tominsups(counts, theta)
    
    
    if id_uci not in ids_uci_special:
        # Correct the minsup values that are too low:
        rminsupsn = [max(x, rmminsup) for x in rminsups]
    else:
        if id_uci == 22:
            rminsupsn = [max(x, 0.08) for x in rminsups]
        elif id_uci == 69:
            rminsupsn = [0,0,0]
            rminsupsn[0] = max(rminsups[0], 0.04)
            rminsupsn[1] = max(rminsups[1], 0.025)
            rminsupsn[2] = max(rminsups[2], 0.01)
        elif id_uci == 26:
            rminsupsn = [max(x, 0.0025) for x in rminsups]
    
    minsupsn = [int(np.ceil(ms * ni)) for ms, ni in zip(rminsupsn, counts)]
    minsupsn = [ms if ms > mminsup else mminsup for ms in minsupsn]
    
    return rminsups, minsups, rminsupsn, minsupsn
    


data_name, X_train, X_test, y_train, y_test, _ = my_utils.get_data_uci(id_uci, encoder='ordinal', seed=seed, X_to_numpy=False)

classes, n_classes = np.unique(y_train, return_counts=True)
    
# Get the minimum chi2 thresold:
dof = len(classes) - 1
theta = getChi2CriticalValue(alpha, dof)

rminsups, minsups, rminsupsn, minsupsn = get_minsups(id_uci, n_classes, theta)

if id_uci not in ids_uci_special:
    minconfs = [minconf] * len(classes)
else:
    if id_uci == 22:
        minconfs = [1.0, 1.0]
    elif id_uci == 69:
        minconfs = [1.0, 1.0, 1.0]
    elif id_uci == 26:
        minconfs = [0.51, 0.9, 1.0]
            
print(data_name)
print('\n##### VALUES FOR THE ENTIRE TRAINING DATASET:')
print(f"Crtitical Value: {theta} - for alpha={alpha} and degrees of freedom={dof}")
print('Training Class Labels:', classes, ' Counts:', n_classes)
print('rminsups:', rminsups, ' minsups:', minsups)
print('rminsupsn:', rminsupsn, ' minsupsn:', minsupsn, '\n')
print('minconfs:', minconfs, '\n')

fminconf = f'{fpath_s}minconfs_{id_uci}_seed{seed}'
np.savetxt(fminconf, np.column_stack((classes, minconfs)), fmt=("%d", "%.2f"))

if nsites > 1:
    skf = StratifiedKFold(n_splits=nsites)
    stratified_partitions = list(skf.split(X_train, y_train))
    
    for ns in range(nsites):
        print(F'SITE {ns} of {nsites-1}')
        
        idxs = stratified_partitions[ns][1]
        X_train_local = X_train[idxs,:]
        y_train_local = y_train[idxs]
        v_l, c_l = np.unique(y_train_local, return_counts=True)
        rminsups, minsups, rminsupsn, minsupsn = get_minsups(id_uci, c_l, theta)
        print('Training Class Labels:', v_l, ' Counts:', c_l)
        print('rminsups:', rminsups, ' minsups:', minsups)
        print('rminsupsn:', rminsupsn, ' minsupsn:', minsupsn, '\n')
        fdata = f'{fpath_s}data_{id_uci}_seed{seed}_s{ns}'
        flabel = f'{fpath_s}label_{id_uci}_seed{seed}_s{ns}'
        fminsup = f'{fpath_s}minsups_{id_uci}_seed{seed}_s{ns}'
        np.savetxt(fdata, X_train_local, delimiter=' ', fmt='%d')
        np.savetxt(flabel, y_train_local, fmt='%d')
        np.savetxt(fminsup, np.column_stack((v_l, minsupsn)), fmt="%d")
        
        # Run RIn-Close_CVCP
        foutput = f'{fpath_s}patterns_{id_uci}_seed{seed}_s{ns}'
        cmd = f'{frinclose} {fdata} cvcp {fminsup} 1 0.0 {foutput} {flabel} {fminconf} 0'
        print('CALLING RINCLOSE...',flush=True)
        subprocess.run(cmd, shell=True, check=True)
        print('')
else:
    fdata = f'{fpath_s}data_{id_uci}_seed{seed}'
    flabel = f'{fpath_s}label_{id_uci}_seed{seed}'
    fminsup = f'{fpath_s}minsups_{id_uci}_seed{seed}'
    np.savetxt(fdata, X_train, delimiter=' ', fmt='%d')
    np.savetxt(flabel, y_train, fmt='%d')
    np.savetxt(fminsup, np.column_stack((classes, minsupsn)), fmt="%d")
    
    # Run RIn-Close_CVCP
    foutput = f'{fpath_s}patterns_{id_uci}_seed{seed}'
    cmd = f'{frinclose} {fdata} cvcp {fminsup} 1 0.0 {foutput} {flabel} {fminconf} 0'
    print('CALLING RINCLOSE...',flush=True)
    subprocess.run(cmd, shell=True, check=True)
