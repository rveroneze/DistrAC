#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 09:39:35 2026

@author: veroneze
"""


import socket
import os
import sys
import collections
import numpy as np
import pandas as pd
import pickle

from utils_PM import getChi2CriticalValue, solve_ties_default_class, get_rankings
from base import CAR_base
from DAC2017 import DAC_ensemble
from AC_consolidation import Consolidation

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


### Set these values:
FBASE_AC = '/Documents/AIDE_results_niid_qs/ACs/'
FBASE_CECI = '/CECI/home/ucl/ingi/rveronez'
consolidation_strategies = ['DAC', 'exact'] #, 'exact_intersection']
alpha = 0.05 # Alpha to be used in the chi2 contingency table
min_conf = 0.51
bytes_per_item = 4
###############################################################################


id_uci = int(sys.argv[1])
n_sites = int(sys.argv[2])
seed = int(sys.argv[3])
alpha_dirichlet = float(sys.argv[4])
print('Dataset:', id_uci, 'n_sites:', n_sites, 'seed:', seed, 'alpha dirichlet:', alpha_dirichlet)

aux = f'nsites{n_sites}/alpha{alpha_dirichlet}/'
FBASE_AC = FBASE_AC + aux
if socket.gethostname() == 'PC-SE24-099':    
    fpath = os.path.expanduser('~'+FBASE_AC)  # Folder to save the results
else:
    fpath = FBASE_CECI + FBASE_AC
print('Folder:')
print(fpath)
print('')

rankings = get_rankings()


# Load datasets
data_name, X_train, X_test, y_train, y_test, _ = my_utils.get_data_uci(id_uci, encoder='no', seed=seed, X_to_numpy=False, stratify=True)

global_classes_train, global_n_classes_train = np.unique(y_train, return_counts=True)
data_classes_counts = dict(zip(global_classes_train, global_n_classes_train))
majority_class = global_classes_train[np.argmax(global_n_classes_train)]
print('Global training Class Labels:', data_classes_counts, ' Majority class:', majority_class)

# Get the minimum chi2 thresold:
number_classes = len(global_classes_train)
dof = number_classes - 1
min_chi2 = getChi2CriticalValue(alpha, dof)
print(f"\nCrtitical Value: {min_chi2} - for alpha={alpha} and degrees of freedom={dof}\n")

# Partition the dataset among the sites (same as previous steps)
clients_data = my_utils.get_partitions_quantity_skew(X_train, y_train, n_sites, seed, alpha=alpha_dirichlet)


# Open the pickle file with the local classifiers:
with open(f"{fpath}local_clfs_{id_uci}_seed{seed}.pkl", "rb") as f:
    results = pickle.load(f)

global_clfs = {}
results_info = []
chaves = [(1, 'select_coverage', 1), (1, 'select_coverage', 2), (1, 'select_coverage', 3), (1, 'select_coverage', 4), (1, 'select_coverage_CBA_M1', None)]
for chave in chaves:
    print(chave)
    
    rank, selection, param = chave
    
    regras_per_site = []
    dict_default_class = collections.defaultdict(int)
    local_datasets = []
    round1_n_rules = 0
    round1_n_items_antecedent = 0
    for i, res in enumerate(results):
        print(f'SITE {i} of {len(results)-1}')
        
        # Load data of site i
        X_train_local, y_train_local = clients_data[i]
        
        if X_train_local.shape[0] < 2:
            print('DATASET HAS LESS THAN 2 RECORDS!')
            continue
        
        # Precompute frozenset representation of instances
        instance_sets = [
            frozenset(instance.items())
            for instance in X_train_local.to_dict(orient="records")
        ]
        
        local_datasets.append( (instance_sets, y_train_local) )
        
        clf = res[chave]
        regras_per_site.append(clf[0])
        default_class_count = clf[1]
        dict_default_class[default_class_count[0]] += default_class_count[1]
        print('Number of rules:', len(clf[0]), 'Default class', default_class_count)
        round1_n_rules += len(clf[0])
        round1_n_items_antecedent += sum([len(rule['antecedent']) for rule in clf[0]])
    print('Total Number of rules:', round1_n_rules, '\n')
    
    for consolidation_strategy in consolidation_strategies:
        print(f'--- {consolidation_strategy} ---')
        
        if consolidation_strategy=='DAC':
            if rank == 2:
                continue
            
            # Communication costs - 1 round only - clients to server
            communication_cost_total = bytes_per_item * round1_n_items_antecedent
            communication_cost_total += bytes_per_item * round1_n_rules
            communication_cost_total += bytes_per_item * round1_n_rules * len(('rsup_rule', 'confidence'))
            communication_cost_total += bytes_per_item * n_sites * 2 # from default_class_count: (class, count)
            
            # DAC Consolidation of the local models
            model = DAC_ensemble(regras_per_site, label_support_metric='rsup_rule')
            global_regras = model.model_consolidation_external()
            
            pm = CAR_base()
            pm.rules = global_regras
            pm.rank_rules(sort_by=rankings[rank][0], ascending=rankings[rank][1])
            
            # Default class label of the classifier:
            default_class = solve_ties_default_class(dict_default_class,
                                            data_classes_counts)
            
            final_n_items_antecedent = sum([len(rule['antecedent']) for rule in pm.rules])
                        
            print('Number of rules after DAC consolidation:', len(global_regras))
            print('Dictionary default class:', dict_default_class)
            print('Default class:', default_class, '\n')
            
            global_clfs[(rank, selection, param, consolidation_strategy)] = (pm.rules, default_class)
            
            results_info.append({
                        'id_uci': id_uci,
                        'nclients': n_sites,
                        'seed': seed,
                        'data_name': data_name,
                        'train_size': X_train.shape[0],
                        'number_classes': number_classes,
                        'majority_class': majority_class,
                        'rank': rank,
                        'selection': selection,
                        'param': param,
                        'consolidation': consolidation_strategy,
                        'round1_n_rules': round1_n_rules,
                        'round1_n_items_antecedent': round1_n_items_antecedent,
                        'round2_n_rules': None,
                        'round2_n_items_antecedent': None,
                        'final_nrules': len(pm.rules),
                        'final_total_items': final_n_items_antecedent,
                        'default_class': default_class,
                        'comm_cost': communication_cost_total
                        })
        else:
            model_consolidation = Consolidation(
                local_classifiers=regras_per_site,
                local_datasets=local_datasets,
                global_classes_count=data_classes_counts,
                exact_count = 'exact' in consolidation_strategy,
                generate_intersections = 'intersection' in consolidation_strategy,
                min_thresholds={'confidence':min_conf, 'chi2':min_chi2},
                ranking_by=rankings[rank],
                )
            rules, default_class = model_consolidation.consolidate()
            print(f'Default class: {default_class} --- {model_consolidation.uncovered_counts}\n')
            
            # Communications costs round 1 (clients to server):
            communication_cost_total = bytes_per_item * round1_n_items_antecedent
            # Communications costs round 2 (server to clients):
            communication_cost_total += bytes_per_item * model_consolidation.round2_n_rules_ # for the rules' id
            communication_cost_total += bytes_per_item * model_consolidation.round2_n_items_antecedent_
            # Communications costs round 3 (clients to server):
            communication_cost_total += bytes_per_item * model_consolidation.round2_n_rules_ # for the rules' id
            communication_cost_total += bytes_per_item * model_consolidation.round2_n_rules_ * number_classes * 2 # *2 because for each label: count
            # Communications costs round 4 (server to clients):
            communication_cost_total += bytes_per_item * model_consolidation.final_n_items_antecedent_
            # Communications costs round 5 (clients to server):
            communication_cost_total += bytes_per_item * number_classes * 2 # *2 because for each label: count
            
            global_clfs[(rank, selection, param, consolidation_strategy)] = (rules, default_class)
            
            results_info.append({
                        'id_uci': id_uci,
                        'nclients': n_sites,
                        'seed': seed,
                        'data_name': data_name,
                        'train_size': X_train.shape[0],
                        'number_classes': number_classes,
                        'majority_class': majority_class,
                        'rank': rank,
                        'selection': selection,
                        'param': param,
                        'consolidation': consolidation_strategy,
                        'round1_n_rules': round1_n_rules,
                        'round1_n_items_antecedent': round1_n_items_antecedent,
                        'round2_n_rules': model_consolidation.round2_n_rules_,
                        'round2_n_items_antecedent': model_consolidation.round2_n_items_antecedent_,
                        'final_nrules': len(rules),
                        'final_total_items': model_consolidation.final_n_items_antecedent_,
                        'default_class': default_class,
                        'comm_cost': communication_cost_total
                        })

df_results_info = pd.DataFrame(data=results_info)
df_results_info.to_csv(f"{fpath}global_clfs_info_{id_uci}_seed{seed}.csv", index=False)
pickle.dump(global_clfs, open(f"{fpath}global_clfs_{id_uci}_seed{seed}.pkl", "wb"))
