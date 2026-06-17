#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 10:03:37 2025

@author: veroneze
"""
import socket
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
import pickle
from utils_PM import getChi2CriticalValue
from base import CAR_base

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


### Set these values:
FBASE_RINCLOSE = '/Documents/AIDE_rinclose/nsites1/'
FBASE_AC = '/Documents/AIDE_res/ACs/nsites1/'
FBASE_CECI = '/CECI/home/ucl/ingi/rveronez/'
alpha = 0.05 # Alpha to be used in the chi2 contingency table
###############################################################################

if socket.gethostname() == 'PC-SE24-099': # Name of my PC
    fpath = os.path.expanduser('~'+FBASE_RINCLOSE)  # Folder with the output of RIn-Close
    fpath_save = os.path.expanduser('~'+FBASE_AC)  # Folder to save the results
else:
    fpath = FBASE_CECI + FBASE_RINCLOSE
    fpath_save = FBASE_CECI + FBASE_AC


id_uci = int(sys.argv[1])
seed = int(sys.argv[2])
print('Dataset:', id_uci, 'seed:', seed)


def predictions(model, X):
    y_pred = model.predict_rule_list(X)
    no_matches = model.no_matches_
    ys_pred = model.predict_rule_set_all(X)
    ys_pred.update({'list': y_pred})
    return ys_pred, no_matches
    
def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    mcc = matthews_corrcoef(y_true, y_pred)
    erros = sum(y_true != y_pred)
    #print(acc, bacc, f1, mcc, erros)
    return {"acc": acc, "bacc": bacc, "f1": f1, 'mcc': mcc, 'erros': erros}

def count_items(rules):
    return sum([len(rule['antecedent']) for rule in rules])

def save_classifier(rules, rank_id, selec_method, param):
    keys_to_save = [key for key in rules[0].keys() if key not in ['covered_rows_antecedent', 'covered_rows_rule']]
    output_filename = f"{fpath_save}rules/id_{id_uci}_seed{seed}_{rank_id}_{selec_method}_{param}.csv"
    my_utils.list_of_dicts_to_file(rules, keys_to_save, output_filename)

print('Loading data')
data_name, X_train, X_test, y_train, y_test, classes = my_utils.get_data_uci(id_uci, encoder='no', seed=seed, X_to_numpy=False)

classes, n_per_label = np.unique(y_train, return_counts=True)
majority_class = classes[np.argmax(n_per_label)]
print('\nTraining Labels: ', classes, n_per_label)

minchi2 = getChi2CriticalValue(alpha=alpha, df=len(classes)-1)
print(f'\nminchi2: {minchi2} for alpha={alpha}')


rankings = {}
rankings[1] = (["confidence", "rsup_rule", "length"], [False, False, True])
rankings[2] = (["confidence", "completeness", "rsup_rule", "length"], [False, False, False, True])
# rankings[3] = (["confidence", "completeness", "rsup_consequent", "length"], [False, False, True, True])


selections = {
    "select_coverage": ("delta", [1, 2, 3, 4]),
    "select_coverage_CBA_M1": (),
}


pm = CAR_base()


print('\nLoading rules from RIn-Close_CVCP output...')
pm.rinclosecvcp_output_to_rules(f"{fpath}patterns_{id_uci}_seed{seed}", X_train, y_train)
total_mined = len(pm.rules)
print('Number of rules: ', total_mined)

#pm.prune_by_thresholds({"rsup_rule": rminsup, "confidence": minconf, "chi2": minchi2})
pm.prune_by_thresholds({"chi2": minchi2})
total_after_pruning = len(pm.rules)
print('\nNumber of patterns after prunning by user-defined thresholds: ', total_after_pruning)

results = []
for rank_id, rank in rankings.items():
    print('\nRank: ', rank_id, rank[0])
    for selec_method, selec_param in selections.items():
        print(f'Selection method: {selec_method}')
        pm.default_class_ = majority_class
        pm.rank_rules(sort_by=rank[0], ascending=rank[1])
        if selec_method == 'none':
            pm.classifier_ = pm.rules
            preds_train, no_matches_train = predictions(pm, X_train)
            preds_test, no_matches_test = predictions(pm, X_test)
            results.append({
                'ranking': rank_id,
                 'selection': selec_method,
                 'param': None,
                 'mined': total_mined,
                 'after_pruning': total_after_pruning,
                 'classifier': len(pm.classifier_),
                 'total_items': None, # I am using None to avoid an expensive and not very informative computation
                 'no_cov_ant': None,
                 'no_cov_rule': None,
                 'no_matches_train': no_matches_train,
                 'no_matches_test': no_matches_test,
                 'preds_train': preds_train,
                 'preds_test': preds_test})
        else:
            method = getattr(pm, selec_method)
            if len(selec_param) == 0:
                method(y_train, sort_by=rank[0], ascending=rank[1])
                preds_train, no_matches_train = predictions(pm, X_train)
                preds_test, no_matches_test = predictions(pm, X_test)
                c_ant, c_rule = pm.get_coverage(pm.classifier_, y_train)
                results.append({
                    'ranking': rank_id,
                    'selection': selec_method,
                    'param': None,
                    'mined': total_mined,
                    'after_pruning': total_after_pruning,
                    'classifier': len(pm.classifier_),
                    'total_items': count_items(pm.classifier_),
                    'no_cov_ant': sum(c_ant == 0),
                    'no_cov_rule': sum(c_rule == 0),
                    'no_matches_train': no_matches_train,
                    'no_matches_test': no_matches_test,
                    'preds_train': preds_train,
                    'preds_test': preds_test})
                save_classifier(pm.classifier_, rank_id, selec_method, '')
            else:
                param = selec_param[0]
                for val in selec_param[1]:
                    print(f"====with {param}={val}...")
                    if selec_method != 'select_top_k_per_class':
                        method(y_train, sort_by=rank[0], ascending=rank[1], **{param: val})
                    else:
                        method(sort_by=rank[0], ascending=rank[1], **{param: val})
                    preds_train, no_matches_train = predictions(pm, X_train)
                    preds_test, no_matches_test = predictions(pm, X_test)
                    c_ant, c_rule = pm.get_coverage(pm.classifier_, y_train)
                    results.append({
                        'ranking': rank_id,
                        'selection': selec_method,
                        'param': val,
                        'mined': total_mined,
                        'after_pruning': total_after_pruning,
                        'classifier': len(pm.classifier_),
                        'total_items': count_items(pm.classifier_),
                        'no_cov_ant': sum(c_ant == 0),
                        'no_cov_rule': sum(c_rule == 0),
                        'no_matches_train': no_matches_train,
                        'no_matches_test': no_matches_test,
                        'preds_train': preds_train,
                        'preds_test': preds_test})
                    save_classifier(pm.classifier_, rank_id, selec_method, val)


my_utils.list_of_dicts_to_file(results, 
                               [key for key in results[0].keys() if key not in ['preds_train', 'preds_test']],
                               f"{fpath_save}info_{id_uci}_seed{seed}.csv")
pickle.dump(results, open(f"{fpath_save}preds_{id_uci}_seed{seed}.pkl", "wb"))


rows = []
for item in results:
    rank_id = item['ranking']
    selec = item['selection']
    param = item['param']
    preds_train = item['preds_train']
    preds_test = item['preds_test']
    
    # Build one row per prediction strategy
    for strategy_name in preds_train.keys():
        row = {
            "ranking": rank_id,
            "selection": selec,
            "param": param,
            "strategy": strategy_name
        }
        
        # compute metrics for training data
        train_metrics = compute_metrics(y_train, preds_train[strategy_name])
        for m_name, m_val in train_metrics.items():
            row[f"{m_name}_train"] = m_val
        
        # compute metrics for test data
        test_metrics = compute_metrics(y_test, preds_test[strategy_name])
        for m_name, m_val in test_metrics.items():
            row[f"{m_name}_test"] = m_val
        
        rows.append(row)

df_performance = pd.DataFrame(rows)
df_performance.to_csv(f"{fpath_save}perf_{id_uci}_seed{seed}.csv", index=False)
