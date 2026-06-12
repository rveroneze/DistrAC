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
ids_uci = [14] # List of datasets' id for which you want to create the script


FOLDER = os.path.expanduser(FOLDER)
alg = 'MLP'
fc = f'--federation-config "options.num-supernodes={SITES}"'
for id_uci in ids_uci:
    with open(f"flwr_{alg}_data{id_uci}_sites{SITES}.sh", "w") as f:
        cabecalho(f)
        
        df = pd.read_csv(FOLDER+'/'+alg+'/central/'+alg+'_central'+str(id_uci)+'.csv')
        for _, row in df.iterrows():
            seed = row['seed']
            id_uci = row['id']
            n_iter = row['n_iter']
            model = row['best_model']
            model = model.replace('\n', '')
            model = model.replace(' ', '')
            model = model.replace("'tanh'", '\\"tanh\\"')
            
            rc = ' --run-config "'
            rc += f'num-server-rounds={n_iter} ' # local_epochs is equal to 1
            rc += f"model-sklearn='{model}' "
            rc += f'id-uci={id_uci} '
            rc += f'seed-random-state={seed}"'
            
            comando = 'flwr run . ' + fc + rc
            f.write(comando)
            f.write('\n')
