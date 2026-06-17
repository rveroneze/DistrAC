#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 15:22:18 2026

@author: veroneze
"""

import socket
import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from utils_PM import getChi2CriticalValue
from base import CAR_base

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


def clean_dict_rules(rules):
    classifier = []
    for r in rules:
        rule = r.copy()
        rule.pop('covered_rows_antecedent') # to save memory
        classifier.append(rule)
    return classifier

### Set these values:
FBASE_RINCLOSE = '/Documents/AIDE_rinclose/'
FBASE_AC = '/Documents/AIDE_res/ACs/'
FBASE_CECI = '/CECI/home/ucl/ingi/rveronez'
alpha = 0.05 # Alpha to be used in the chi2 contingency table
###############################################################################


id_uci = int(sys.argv[1])
n_sites = int(sys.argv[2])
seed = int(sys.argv[3])
print('Dataset:', id_uci, 'Nsites:', n_sites, 'seed:', seed)

rankings = {}
rankings[1] = (["confidence", "rsup_rule", "length"], [False, False, True])
rankings[2] = (["confidence", "completeness", "rsup_rule", "length"], [False, False, False, True])

selections = {
    "select_coverage": ("delta", [1, 2, 3, 4]),
    "select_coverage_CBA_M1": (),
}


FBASE_RINCLOSE = FBASE_RINCLOSE + 'nsites' + str(n_sites) + '/'
FBASE_AC = FBASE_AC + 'nsites' + str(n_sites) + '/'
if socket.gethostname() == 'PC-SE24-099':    
    fpath = os.path.expanduser('~'+FBASE_RINCLOSE)  # Folder with the output of RIn-Close
    fpath_save = os.path.expanduser('~'+FBASE_AC)  # Folder to save the results
else:
    fpath = FBASE_CECI + FBASE_RINCLOSE
    fpath_save = FBASE_CECI + FBASE_AC
print('Folders:')
print(fpath)
print(fpath_save)
print('')


_, X_train, _, y_train, _, classes = my_utils.get_data_uci(id_uci, encoder='no', seed=seed, X_to_numpy=False)
skf = StratifiedKFold(n_splits=n_sites)
stratified_partitions = list(skf.split(X_train, y_train))

results = []
for ns in range(n_sites):
    print(F'SITE {ns} of {n_sites-1}')
    
    print('Getting local data')
    idxs = stratified_partitions[ns][1]
    X_train_local = X_train.iloc[idxs,:]
    y_train_local = y_train[idxs]
    classes_local, n_per_label_local = np.unique(y_train_local, return_counts=True)
    
    minchi2 = getChi2CriticalValue(alpha=alpha, df=len(classes_local)-1)
    print(f'Local minchi2: {minchi2} for alpha={alpha}')
    
    pm = CAR_base()
    
    print('\nLoading rules from RIn-Close_CVCP output...')
    pm.rinclosecvcp_output_to_rules(f"{fpath}patterns_{id_uci}_seed{seed}_s{ns}", X_train_local, y_train_local)
    total_mined = len(pm.rules)
    print('Number of rules: ', total_mined)
    
    pm.prune_by_thresholds({"chi2": minchi2})
    total_after_pruning = len(pm.rules)
    print('\nNumber of patterns after prunning by user-defined thresholds: ', total_after_pruning)
    
    
    print('Building classifiers...')
    local_results = {}
    for rank_id, rank in rankings.items():
        print('\nRank: ', rank_id, rank[0])
        for selec_method, selec_param in selections.items():
            print(f'Selection method: {selec_method}')
            
            method = getattr(pm, selec_method)
            if len(selec_param) == 0:
                method(y_train_local, sort_by=rank[0], ascending=rank[1])
                print('Number of rules in the classifier: ', len(pm.classifier_))
                local_results[(rank_id, selec_method, None)] = (
                    clean_dict_rules(pm.classifier_),
                    (pm.default_class_, pm.notCoveredRows_)
                    )
            else:
                param = selec_param[0]
                for val in selec_param[1]:
                    print(f"====with {param}={val}...")
                    method(y_train_local, sort_by=rank[0], ascending=rank[1], **{param: val})
                    print('Number of rules in the classifier: ', len(pm.classifier_))
                    local_results[(rank_id, selec_method, val)] = (
                        clean_dict_rules(pm.classifier_),
                        (pm.default_class_, pm.notCoveredRows_)
                        )
                    
    results.append(local_results)
    print('')

pickle.dump(results, open(f"{fpath_save}local_clfs_{id_uci}_seed{seed}.pkl", "wb"))
