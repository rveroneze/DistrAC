#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:29:42 2025

@author: veroneze
"""

# Load Last Model Checkpoint

import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn1.task import get_model, set_model_params


# # # NEEDS TO RUN THAT BEFORE OUTSIDE HERE or do the EXPORT:
# # import sys
# # sys.path.append(os.path.expanduser("~/Documents/github/AIDE/codes"))

# # ###############################################################################
# # Tells Python where your project root is for your current terminal session
# # It allows imports to work without complex build files.
# # On Linux/macOS, run the command:
# # export PYTHONPATH=/home/veroneze/Documents/github/AIDE/codes
# # => Replace /path/to/your/project/root with your actual path
# from common.my_utils import get_data_uci
# # ###############################################################################


###############################################################################
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_codes = os.path.abspath(os.path.join(current_script_dir, '..'))
sys.path.append(project_codes)
from common import my_utils
###############################################################################


# Set these values:
FOLDER = '~/Documents/AIDE_res/sklearn' 
ALG = 'LR'
ID_UCI = 14
SEED = 138
NSITES = 10


FOLDER = os.path.expanduser(FOLDER)
FPATH = f'{FOLDER}/{ALG}/nsites{NSITES}/{ALG}_'

# Load dataset
_, _, X_test, _, y_test, classes = my_utils.get_data_uci(ID_UCI, seed=SEED)

# Load best model in the centralized training
if ALG == 'MLP':
    df_central = pd.read_csv(FOLDER+'/'+ALG+'/central/'+ALG+'_central'+str(ID_UCI)+'.csv', index_col=0)
    best_model = df_central.at[SEED, "best_model"]
    best_model = best_model.replace('\n', '')
    best_model = best_model.replace(' ', '')
    print(best_model)
else:
    best_model = "LogisticRegression()"


# Load model's parameters
list_of_files = [fname for fname in glob.glob(FPATH+"DATA"+str(ID_UCI)+"_SEED"+str(SEED)+"_round*params.npz")]
latest_round_file = max(list_of_files, key=os.path.getctime)
print("Loading pre-trained model from: ", latest_round_file)

npzfile = np.load(latest_round_file)
print(npzfile.files)

params_fl = [npzfile[f] for f in npzfile.files]


# Create the model and set its parameters
model = get_model(best_model, 1, classes, X_test.shape[1])
set_model_params(model, params_fl)

# Get predictions and scores
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
bacc = balanced_accuracy_score(y_test, y_pred)

print("Test accuracy:", acc)
print("Test b. accuracy:", bacc)