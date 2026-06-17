#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 09:52:06 2025

@author: veroneze
"""

import socket
import os
import sys
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from utils_PM import  boolDF2Transactions, getChi2CriticalValue, get_coverage
from DAC2017 import DAC, DAC_ensemble

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################

# Set Folder to save the results
FBASE_DAC = '/Documents/AIDE_res/DAC/'
FBASE_CECI = '/CECI/home/ucl/ingi/rveronez'
# -----------------------------------------------------------------------------


id_uci = int(sys.argv[1])
nclients = int(sys.argv[2])
print('Dataset:', id_uci, 'n_clients:', nclients)


if socket.gethostname() == 'PC-SE24-099':    
    fpath = os.path.expanduser('~'+FBASE_DAC)  # Folder to save the results
else:
    fpath = FBASE_CECI + FBASE_DAC
print('Folder:')
print(fpath)
print('')


min_support_threshold = 0.0002
min_confidence_threshold = 0.51
alpha = 0.05 # Alpha to be used in the chi2 contingency table


results = []
for seed in my_utils.get_seeds():
    print('seed:', seed)
    data_name, X_train, X_test, y_train, y_test, classes = my_utils.get_data_uci(id_uci, seed=seed, X_to_numpy=False)
    print(id_uci, data_name, '\n')
    
    sorted_classes = sorted(list(classes))
    
    min_chi2_threshold = getChi2CriticalValue(alpha=alpha, df=len(classes)-1)
    
    trans_train = boolDF2Transactions(X_train)
    trans_test = boolDF2Transactions(X_test)
    
    communication_cost = 0
    if nclients > 1:
        skf = StratifiedKFold(n_splits=nclients)
        stratified_partitions = list(skf.split(X_train, y_train))
        
        models = []
        for nc in range(nclients):
            print("Training client: ", nc)
            
            idxs = stratified_partitions[nc][1]
            trans_train_nc = boolDF2Transactions(X_train.iloc[idxs,:])
            y_train_nc = y_train[idxs]
            
            model_nc = DAC(min_sup=min_support_threshold, min_conf=min_confidence_threshold, min_chi2=min_chi2_threshold)
            model_nc.fit(trans_train_nc, y_train_nc)
            models.append(model_nc)
            communication_cost += sum([4*len(rule['antecedent'])+16 for rule in model_nc.rules_])
        
        model = DAC_ensemble(models)
        model.model_consolidation()
        rules = model.get_rules()
        classes = model.get_classes()
    else:
        model = DAC(min_sup=min_support_threshold, min_conf=min_confidence_threshold, min_chi2=min_chi2_threshold)
        model.fit(trans_train, y_train)
        rules = model.rules_
        classes = model.classes_
        
    proba_train, no_matches_train = model.predict_proba(trans_train)
    y_pred_train, _ = model.predict(trans_train)
    loss_train = log_loss(y_train, proba_train, labels=sorted_classes)
    acc_train = accuracy_score(y_train,  y_pred_train)
    bacc_train = balanced_accuracy_score(y_train,  y_pred_train)
    erros_train = sum(y_train != y_pred_train)
    
    proba_test, no_matches_test = model.predict_proba(trans_test)
    y_pred_test, _ = model.predict(trans_test)
    loss_test = log_loss(y_test, proba_test, labels=sorted_classes)
    acc_test = accuracy_score(y_test,  y_pred_test)
    bacc_test = balanced_accuracy_score(y_test,  y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, average='macro')
    mcc_test = matthews_corrcoef(y_test, y_pred_test)
    erros_test = sum(y_test != y_pred_test)
    
    c_ant, c_rule = get_coverage(rules, trans_train, y_train)
    no_cov_ant = sum(c_ant == 0)
    no_cov_rule = sum(c_rule == 0)
    total_items = sum([len(rule['antecedent']) for rule in rules])
    
    elemento = {'id_uci': id_uci,
                'nclients': nclients,
                'seed': seed,
                'data_name': data_name,
                'train_size': X_train.shape[0],
                'nrules': len(rules),
                'total_items': total_items,
                'no_cov_ant': no_cov_ant,
                'no_cov_rule': no_cov_rule,
                'loss_train': loss_train,
                'no_matches_train': no_matches_train,
                'acc_train': acc_train,
                'bacc_train': bacc_train,
                'erros_train': erros_train,
                'loss_test': loss_test,
                'no_matches_test': no_matches_test,
                'acc_test': acc_test,
                'bacc_test': bacc_test,
                'f1_test': f1_test,
                'mcc_test': mcc_test,
                'erros_test': erros_test,
                'comm_cost': communication_cost}
    results.append(elemento)
    
    pd.DataFrame(rules).to_csv(f"{fpath}DAC_rules_{id_uci}_seed{seed}_nc{nclients}.csv", sep=',', index=False)
    #pd.DataFrame(y_pred_train).to_csv(f"{fpath}DAC_ytrain_{id_uci}_seed{seed}_nc{nclients}.csv", index=False, header=False)
    #pd.DataFrame(y_pred_test).to_csv(f"{fpath}DAC_ytest_{id_uci}_seed{seed}_nc{nclients}.csv", index=False, header=False)

pd.DataFrame(results).to_csv(f"{fpath}DAC_{id_uci}_nc{nclients}.csv", sep=',', index=False, mode='w')
