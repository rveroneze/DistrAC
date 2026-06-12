#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 17:28:25 2025

@author: veroneze
"""

import re
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import OrdinalEncoder
import csv
import pickle

# missing value representation used by RIn-Close
MV_RINCLOSE = 999999


def get_ids_uci():
    # ids_uci_nused = [12, 83, 90] # Balance Scale, Primary Tumor, Soybean (Large)
    return [26, 76, 73, 22, 69, 19, 101, 936, 105, 70, 14]

def get_seeds():
    return [5, 14, 17, 19, 29, 31, 34, 42, 49, 63, 64, 65, 68, 74, 76, 100, 114, 133, 138, 148]


def preprocess_rinclosecvcp_with_not_equal(X0: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a categorical dataframe according to the following rules:
    - If a column has exactly 2 distinct non-NaN values → keep as is (replace NaN with mv_value)
    - If a column has 3 or more distinct non-NaN values → one-hot encode it (ignoring NaNs)
    - If a column has only one unique non-NaN value OR is entirely NaN → skip it
    - Otherwise (e.g., mix of single value + NaNs) → keep the column, replacing NaN with mv_value
    It will allow having the condition "is not equal to" in addition to "is equal to"
    
    Parameters
    ----------
    X0 : pd.DataFrame
        Input dataframe with only categorical attributes.
        
    Returns
    -------
    X : pd.DataFrame
        Processed dataframe according to the rules above.
    """
    
    enc = OrdinalEncoder(encoded_missing_value=MV_RINCLOSE, dtype=np.int32)
    
    X_parts = []
    for col in X0.columns:
        unique_vals = X0[col].dropna().unique()
        n_unique = len(unique_vals)

        # Case 1: Binary categorical attribute
        if n_unique == 2:
            X_col = enc.fit_transform(X0[[col]])
            X_col = pd.DataFrame(X_col, columns=[col], index=X0.index)
            X_parts.append(X_col)
            
        # Case 2: Multi-class categorical attribute
        elif n_unique >= 3:
            X_dummies = pd.get_dummies(X0[col], prefix=col, dummy_na=False)
            X_parts.append(X_dummies.astype(int))
            
        # Case 3: Single-value
        elif n_unique == 1:
            # Check if all values equal (no variability) or we have NaN values
            if X0[col].isna().sum() > 0:
                X_col = enc.fit_transform(X0[[col]])
                X_col = pd.DataFrame(X_col, columns=[col], index=X0.index)
                X_parts.append(X_col)
                
    X = pd.concat(X_parts, axis=1)
    return X


def preprocess_one_hot_expansion(X0: pd.DataFrame) -> pd.DataFrame:
    """
    This method goes beyond standard one-hot encoding by also including 
    “not equal” binary features when a categorical attribute has ≥ 3 unique values
    """
    X_new = pd.DataFrame(index=X0.index)
    
    for col in X0.columns:
        values = X0[col].unique()
        n_values = len(values)

        for val in values:
            # always add == feature
            eq_col = f"{col}=={val}"
            X_new[eq_col] = (X0[col] == val).astype(int)

            # add != feature only if there are >=3 distinct values
            if n_values >= 3:
                neq_col = f"{col}!={val}"
                X_new[neq_col] = (X0[col] != val).astype(int)

    return X_new



def get_data_uci(id_uci, encoder='one-hot', seed=42, X_to_numpy=True, stratify=False):
    #dataset = fetch_ucirepo(id=id_uci)
    """
    Sometimes, I am facing problems of connection with UCI repository.
    So, I saved the datasets.
    """
    with open(f"../../datasets/uci_{id_uci}.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    X0 = dataset.data.features
    y = dataset.data.targets
    
    # Assures all non-null values are strings while keeping NaNs:
    X0 = X0.apply(lambda col: col.map(lambda x: str(x) if pd.notnull(x) else x))
    
    if encoder=='one-hot':
        # Convert categorical variable into dummy/indicator variables (one-hot enconding):
        X = pd.get_dummies(X0)
        y = y.to_numpy().ravel()
    elif encoder=='ordinal':
        # Encode categorical features as an integer array:
        enc = OrdinalEncoder(encoded_missing_value=MV_RINCLOSE, dtype=np.int32)
        enc.fit(X0)
        X = enc.transform(X0)
        y = enc.fit_transform(y).ravel()
    else:
        X = X0
        y = y.to_numpy().ravel()
    if X_to_numpy:
        X = X.to_numpy()
    
    if not stratify:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    classes = np.unique(y_train)
    
    # sklearn>=1.7.0 (https://scikit-learn.org/stable/whats_new/v1.7.html#version-1-7-0):
    # log_loss now raises a ValueError if values of y_true are missing in the parameter labels
    # So, I am removing these instances from X_test and y_test
    mask = np.isin(y_test, classes) # boolean mask: True where y_test is in y_train
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]

    return dataset.metadata['name'], X_train, X_test_filtered, y_train, y_test_filtered, classes


def list_of_dicts_to_file(list_of_dicts, keys_to_save, output_filename):
    try:
        with open(output_filename, 'w', newline='', encoding='utf-8') as output_file:
            # Create a DictWriter object
            writer = csv.DictWriter(output_file, fieldnames=keys_to_save, extrasaction='ignore')
            
            # Write the header row
            writer.writeheader()
            
            # Write all data rows at once
            writer.writerows(list_of_dicts)
    except Exception as e:
        print(f"An error occurred: {e}")


def get_scores_sklearn(model, X, y):
    # Log loss, aka logistic loss or cross-entropy loss:
    loss = log_loss(y, model.predict_proba(X), labels=model.classes_)
    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    bacc = float(balanced_accuracy_score(y, y_pred))
    f1 = f1_score(y, y_pred, average='macro')
    mcc = matthews_corrcoef(y, y_pred)
    return [y_pred, loss, acc, bacc, f1, mcc]

def extract_history(filepath):
    centralized_losses = []
    acc_lines = []
    bacc_lines = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_centralized_loss = False
    in_acc = False
    in_bacc = False

    for line in lines:
        # Clean the line by removing the timestamp and log level
        if 'INFO:' in line:
            line = line.split('INFO:')[1].strip()

        # Check section switches
        if line.startswith("History (loss, centralized):"):
            in_centralized_loss = True
            in_acc = in_bacc = False
            continue
        elif line.startswith("{'acc':"):
            in_acc = True
            in_centralized_loss = in_bacc = False
            acc_lines.append(line)
            continue
        elif line.startswith("'bacc':"):
            in_bacc = True
            in_acc = in_centralized_loss = False
            bacc_lines.append(line)
            continue

        # Add lines to appropriate sections
        if in_centralized_loss:
            match_ok = re.search(r'round \d+: ([0-9.]+)', line)
            if match_ok:
                centralized_losses.append(float(match_ok.group(1)))
        elif in_acc:
            acc_lines.append(line)
        elif in_bacc:
            bacc_lines.append(line)
            if line.endswith(')]}'):
                in_bacc = False

    # Build full string for eval
    acc_str = ''.join(acc_lines)
    bacc_str = ''.join(bacc_lines)
    
    # Add necessary brackets if missing
    if not acc_str.endswith("}"):
        acc_str += "}"
    if not bacc_str.startswith("{"):
        bacc_str = "{" + bacc_str
    
    # Evaluate
    acc = [val for _, val in ast.literal_eval(acc_str)['acc']]
    bacc = [val for _, val in ast.literal_eval(bacc_str)['bacc']]
    
    return centralized_losses, acc, bacc


def get_data_uci(id_uci, encoder='one-hot', seed=42, X_to_numpy=True):
    #dataset = fetch_ucirepo(id=id_uci)
    """
    Sometimes, I am facing problems of connection with UCI repository.
    So, I saved the datasets.
    """
    with open(f"../../datasets/uci_{id_uci}.pkl", "rb") as f:
        dataset = pickle.load(f)
    
    X0 = dataset.data.features
    y = dataset.data.targets
    
    # Assures all non-null values are strings while keeping NaNs:
    X0 = X0.apply(lambda col: col.map(lambda x: str(x) if pd.notnull(x) else x))
    
    if encoder=='one-hot':
        # Convert categorical variable into dummy/indicator variables (one-hot enconding):
        X = pd.get_dummies(X0)
        y = y.to_numpy().ravel()
    elif encoder=='ordinal':
        # Encode categorical features as an integer array:
        enc = OrdinalEncoder(encoded_missing_value=MV_RINCLOSE, dtype=np.int32)
        enc.fit(X0)
        X = enc.transform(X0)
        y = enc.fit_transform(y).ravel()
    else:
        X = X0
        y = y.to_numpy().ravel()
    if X_to_numpy:
        X = X.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, shuffle=True, stratify=y)
    classes = np.unique(y_train)
    
    # sklearn>=1.7.0 (https://scikit-learn.org/stable/whats_new/v1.7.html#version-1-7-0):
    # log_loss now raises a ValueError if values of y_true are missing in the parameter labels
    # So, I am removing these instances from X_test and y_test
    mask = np.isin(y_test, classes) # boolean mask: True where y_test is in y_train
    X_test_filtered = X_test[mask]
    y_test_filtered = y_test[mask]

    return dataset.metadata['name'], X_train, X_test_filtered, y_train, y_test_filtered, classes


# Data Quantity Skew
def get_partitions_quantity_skew(X_train, y_train, nsites, seed, alpha=0.05):
    rng = np.random.default_rng(seed)
    
    N = len(X_train)
    
    # Generate unequal client sizes
    proportions = rng.dirichlet(np.ones(nsites) * alpha)
    
    client_sizes = (proportions * N).astype(int)
    
    # Adjust rounding
    client_sizes[-1] += N - client_sizes.sum()
    
    # Shuffle data
    idxs = rng.permutation(N)
    
    # Detect input types
    X_is_df = isinstance(X_train, pd.DataFrame)
    y_is_series = isinstance(y_train, pd.Series)
        
    clients_data = []
    start = 0
    for size in client_sizes:
        client_idxs = idxs[start:start+size]
        
        if X_is_df:
            X_local = X_train.iloc[client_idxs].reset_index(drop=True)
        else:
            X_local = X_train[client_idxs]
        
        if y_is_series:
            y_local = y_train.iloc[client_idxs].reset_index(drop=True)
        else:
            y_local = y_train[client_idxs]
    
        clients_data.append((X_local, y_local))
    
        start += size
    
    return clients_data


def get_dataset_name(id_uci):
    with open(f"../../datasets/uci_{id_uci}.pkl", "rb") as f:
        dataset = pickle.load(f)
    nome = dataset.metadata['name']
    stops = [r'\s\(', r'\son\b', r'\sRecords\b', r'\sEndgame\b']
    pattern = rf'^(.*?)(?={"|".join(stops)}|$)'
    return re.search(pattern, nome).group(1)