#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 19:12:08 2025

@author: veroneze
"""

import os
import pandas as pd

def cabecalho(file):
    file.write("#!/bin/bash")
    file.write('\n')
    file.write('export PYTHONPATH=$(dirname "$PWD")')
    file.write('\n\n')

# Set Folder with the results and the number of sites:
FOLDER = '~/Documents/AIDE_res/sklearn/'
SITES = 10

FOLDER = os.path.expanduser(FOLDER)
alg = 'LR'
fc = f'--federation-config "options.num-supernodes={SITES}"'
with open(f"flwr_{alg}_sites{SITES}.sh", "w") as f:
    cabecalho(f)
    
    df = pd.read_csv(FOLDER+'/'+alg+'/central/'+alg+'_central.csv')
    for _, row in df.iterrows():
        seed = row['seed']
        id_uci = row['id']
        n_iter = row['n_iter']
        model = "'LogisticRegression()'"
                
        rc = ' --run-config "'
        rc += f'num-server-rounds={n_iter} ' # local_epochs is equal to 1
        rc += f'model-sklearn={model} '
        rc += f'id-uci={id_uci} '
        rc += f'seed-random-state={seed}"'
        
        comando = 'flwr run . ' + fc + rc
        f.write(comando)
        f.write('\n')
