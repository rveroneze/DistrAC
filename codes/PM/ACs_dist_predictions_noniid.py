#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 19:34:09 2026

@author: veroneze
"""

import socket
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
import pickle

from base import CAR_base

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


### Set these values:
FBASE_AC = '/Documents/AIDE_results_niid_qs/ACs/'
FBASE_CECI = '/CECI/home/ucl/ingi/rveronez'
###############################################################################


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    erros = sum(y_true != y_pred)
    # print(acc, bacc, f1, mcc, erros)
    return {"acc": acc, "bacc": bacc, "f1": f1, 'mcc': mcc, 'erros': erros}


id_uci = int(sys.argv[1])
n_sites = int(sys.argv[2])
alpha_dirichlet = float(sys.argv[3])
print('Dataset:', id_uci, 'Nsites:', n_sites, 'alpha dirichlet:', alpha_dirichlet)

aux = f'nsites{n_sites}/alpha{alpha_dirichlet}/'
FBASE_AC = FBASE_AC + aux
if socket.gethostname() == 'PC-SE24-099':    
    fpath = os.path.expanduser('~'+FBASE_AC)  # Folder to save the results
else:
    fpath = FBASE_CECI + FBASE_AC
print('Folder:')
print(fpath)
print('')

results = []
seeds = my_utils.get_seeds()
for seed in seeds:
    print('Seed: ', seed)
    
    # Load datasets
    _, X_train, X_test, y_train, y_test, _ = my_utils.get_data_uci(id_uci, encoder='no', seed=seed, X_to_numpy=False, stratify=True)
    
    global_classes_train, global_n_classes_train = np.unique(y_train, return_counts=True)
    print('Global training Class Labels:', global_classes_train, ' Counts:', global_n_classes_train)
    
    majority_class = global_classes_train[np.argmax(global_n_classes_train)]
    print("Majority class in training data:", majority_class)
    
    # Open the pickle file with the local classifiers:
    with open(f"{fpath}global_clfs_{id_uci}_seed{seed}.pkl", "rb") as f:
        global_clfs = pickle.load(f)
    
    for chave, clf in global_clfs.items():
        print(chave)
        rank, selection, param, consolidation_strategy = chave
        
        print('Number of rules:', len(clf[0]), 'Default class', clf[1])
        
        pm = CAR_base()
        pm.rules = clf[0]
        pm.default_class_ = clf[1]
        
        y_pred = pm.predict_rule_list(X_test)
        no_matches = pm.no_matches_
        test_metrics = compute_metrics(y_test, y_pred)
        
        results.append({
            'seed': seed,
            'ranking': rank,
            'selection': selection,
            'param': param,
            'consolidation': consolidation_strategy,
            'strategy': 'list',
            'majority_class': majority_class,
            'no_matches_test': no_matches,
            'acc_test': test_metrics['acc'],
            'bacc_test': test_metrics['bacc'],
            'f1_test': test_metrics['f1'],
            'mcc_test': test_metrics['mcc'],
            'erros_test': test_metrics['erros'],
            })
        
df_results = pd.DataFrame(data=results)
df_results.to_csv(f"{fpath}global_clfs_perf_{id_uci}.csv", index=False)
