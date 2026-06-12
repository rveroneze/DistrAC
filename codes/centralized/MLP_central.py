#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:52:23 2025

@author: veroneze
"""

# https://scikit-learn.org/stable/modules/grid_search.html#grid-search
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html

import os
import sys
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


# Set Folder to save the results and maximum number of sites used in the experiments: 
FOLDER = '~/Documents/AIDE_res/sklearn/MLP/central/'
max_sites = 10



id_uci = int(sys.argv[1])
print('Dataset:', id_uci)


BASE_DIR = os.path.expanduser(FOLDER)
seeds = my_utils.get_seeds()

results = []
col_names = ['seed',
             'id',
             'name',
             'best_model',
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
for seed in seeds:
    data_name, X_train, X_test, y_train, y_test, classes = my_utils.get_data_uci(id_uci, seed=seed)
    print(id_uci, data_name)
    
    m = X_train.shape[1]
    n_neurons = [int(np.rint(m * factor)) for factor in np.arange(0.25, 1.75, 0.25)]
    hl1 = [(n,) for n in n_neurons]
    hl2 = [(n, int(np.rint(n/2))) for n in n_neurons[1:]]
    
    p_grid = {'activation': ['relu', 'tanh'], 'hidden_layer_sizes': hl1 + hl2}
    
    bs = min(64, int(np.floor(X_train.shape[0] / max_sites)))
    model = MLPClassifier(max_iter=5000, batch_size=bs, random_state=42)
    clf = GridSearchCV(estimator=model, param_grid=p_grid)
    clf.fit(X_train, y_train)
    
    best_model = clf.best_estimator_
    best_model.fit(X_train, y_train)
    scores_train = my_utils.get_scores_sklearn(best_model, X_train, y_train)
    scores_test = my_utils.get_scores_sklearn(best_model, X_test, y_test)
    
    y_pred = scores_test[0]
    
    elemento = [seed,
                id_uci,
                data_name,
                best_model,
                len(best_model.loss_curve_),
                ] + scores_train[1:] + scores_test[1:]
    results.append(elemento)
    
    pd.DataFrame(y_pred).to_csv(f"{BASE_DIR}MLP_central{id_uci}_seed{seed}.csv", index=False, header=False)

df = pd.DataFrame(results, columns=col_names)
df.to_csv(f'{BASE_DIR}MLP_central{id_uci}.csv', sep=',', index=False)
