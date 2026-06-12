"""sklearn1: A Flower / sklearn app."""

import warnings
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ###############################################################################
# Tells Python where your project root is for your current terminal session
# It allows imports to work without complex build files.
# On Linux/macOS, run the command:
# export PYTHONPATH=/home/veroneze/Documents/github/AIDE/codes
# => Replace /path/to/your/project/root with your actual path
from common.my_utils import get_data_uci
# ###############################################################################
#from sklearn1.my_utils import get_data_uci


# Global cache
full_dataset = None  # Cache FederatedDataset
stratified_partitions = None

def load_data(id_uci: int, partition_id: int, num_partitions: int, seed: int):
    """Load partition data."""
    # Only initialize `FederatedDataset` once
    global full_dataset, stratified_partitions
    if full_dataset is None:
        # fetch dataset
        _, X_train, X_test, y_train, y_test, classes = get_data_uci(id_uci, seed=seed)
        full_dataset = (X_train, X_test, y_train, y_test, classes)
        
        # Generate stratified partitions (once)
        skf = StratifiedKFold(n_splits=num_partitions)
        stratified_partitions = list(skf.split(X_train, y_train))
    
    X_train, X_test, y_train, y_test, classes = full_dataset
    
    # Manual partitioning based on IID and shuffle (shuffle done in get_data_uci)
    idxs = stratified_partitions[partition_id][1]
    X_train_local, y_train_local = X_train[idxs,:], y_train[idxs]
    
    return X_train_local, X_test, y_train_local, y_test, classes


def get_model(model_sklearn: str, local_epochs: int, classes: np.ndarray, n_features: int):
    model = eval(model_sklearn)
    model.set_params(max_iter=local_epochs, random_state=42, warm_start=True)
    np.random.seed(42)
    X_dummy = np.random.randint(2, size=(classes.size, n_features))
    
    # Ignore convergence failure due to low local epochs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_dummy, classes) # MLP needs this dummy training
    return model


def get_model_params(model):
    if model.__class__.__name__ == 'LogisticRegression':
        if model.fit_intercept:
            params = [model.coef_, model.intercept_]
        else:
            params = [model.coef_]
    else: #MLP
        coefs = [coef for coef in model.coefs_]
        intercepts = [intercept for intercept in model.intercepts_]
        params = coefs + intercepts
    return params


def set_model_params(model, params):
    if model.__class__.__name__ == 'LogisticRegression':
        model.coef_ = params[0]
        if model.fit_intercept:
            model.intercept_ = params[1]
    else: #MLP
        model.coefs_ = [matriz for matriz in params if len(matriz.shape)>1]
        model.intercepts_ = [matriz for matriz in params if len(matriz.shape)<2]
    return model


# def set_initial_params(model, n_classes, n_features):
#     model.coef_ = np.zeros((n_classes, n_features))
#     if model.fit_intercept:
#         model.intercept_ = np.zeros((n_classes,))


def align_model_params_2client(model_global, model_local, indices_to_keep):
    if model_global.__class__.__name__ == 'LogisticRegression':
        model_local.coef_ = model_global.coef_[indices_to_keep, :]
        if model_local.fit_intercept:
            model_local.intercept_ = model_global.intercept_[indices_to_keep]
    else: #MLP
        model_local.coefs_[-1] = model_global.coefs_[-1][:, indices_to_keep]
        model_local.intercepts_[-1] = model_global.intercepts_[-1][indices_to_keep]
    return model_local

def align_model_params_2server(model_global, model_local, indices_to_update):
    if model_global.__class__.__name__ == 'LogisticRegression':
        model_global.coef_[indices_to_update, :] = model_local.coef_
        if model_local.fit_intercept:
            model_global.intercept_[indices_to_update] = model_local.intercept_
    else: #MLP
        model_global.coefs_[-1][:, indices_to_update] = model_local.coefs_[-1]
        model_global.intercepts_[-1][indices_to_update] = model_local.intercepts_[-1]
    return model_global
