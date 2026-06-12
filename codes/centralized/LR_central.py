#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 15:17:35 2025

@author: veroneze
"""

import os
import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


# Set Folder to save the results: 
FOLDER = '~/Documents/AIDE_res/sklearn/LR/central/'



BASE_DIR = os.path.expanduser(FOLDER)
seeds = my_utils.get_seeds()
ids_uci = my_utils.get_ids_uci()

results = []
col_names = ['seed',
             'id',
             'name',
             'n_iter',
             'loss_train',
             'acc_train',
             'bacc_train',
             'f1_train',
             'mcc_train',
             'loss_test',
             'acc_test',
             'bacc_test',
             'f1_test',
             'mcc_test',
    ]
for id_uci in ids_uci:
    for seed in seeds:
        data_name, X_train, X_test, y_train, y_test, classes = my_utils.get_data_uci(id_uci, seed=seed)
        print(id_uci, data_name, X_train.shape)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        print('Number of iterations: ', model.n_iter_)
        
        scores_train = my_utils.get_scores_sklearn(model, X_train, y_train)
        scores_test = my_utils.get_scores_sklearn(model, X_test, y_test)
        
        y_pred = scores_test[0]
        
        elemento = [seed,
                    id_uci,
                    data_name,
                    model.n_iter_[-1],
                    ] + scores_train[1:] + scores_test[1:]
        results.append(elemento)
        
        pd.DataFrame(y_pred).to_csv(f"{BASE_DIR}LR_central{id_uci}_seed{seed}.csv", index=False, header=False)

df = pd.DataFrame(results, columns=col_names)
df.to_csv(f'{BASE_DIR}LR_central.csv', sep=',', index=False)
