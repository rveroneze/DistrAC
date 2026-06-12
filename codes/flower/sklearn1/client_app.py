"""sklearn1: A Flower / sklearn app."""

import warnings

import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from sklearn1.task import (
    load_data,
    get_model,
    set_model_params,
    get_model_params,
    align_model_params_2client,
    align_model_params_2server,
)

# ###############################################################################
# Tells Python where your project root is for your current terminal session
# It allows imports to work without complex build files.
# On Linux/macOS, run the command:
# export PYTHONPATH=/home/veroneze/Documents/github/AIDE/codes
# => Replace /path/to/your/project/root with your actual path
from common.my_utils import get_scores_sklearn
# ###############################################################################
#from sklearn1.my_utils import get_scores_sklearn

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, model, X_train, X_test, y_train, y_test):
        self.partition_id = partition_id
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.classes_train = np.unique(y_train)
        self.model_local = None
        self.align_params = len(self.model.classes_) != len(self.classes_train)
        if self.align_params:
            self.model_local = get_model(repr(model), model.max_iter, self.classes_train, X_train.shape[1])
            self.idx_labels = [i for i, cls in enumerate(self.model.classes_) if cls in self.classes_train]
    
    def fit(self, parameters, config):
        #print(f"[Client {self.partition_id}] FIT, config: {config}")
        set_model_params(self.model, parameters)
        
        if not self.align_params:
            with warnings.catch_warnings(): # Ignore convergence failure due to low local epochs
                warnings.simplefilter("ignore")
                self.model.fit(self.X_train, self.y_train)
        else:
            self.model_local = align_model_params_2client(self.model, self.model_local, self.idx_labels)
            
            with warnings.catch_warnings(): # Ignore convergence failure due to low local epochs
                warnings.simplefilter("ignore")
                self.model_local.fit(self.X_train, self.y_train)
            
            self.model = align_model_params_2server(self.model, self.model_local, self.idx_labels)
            
        return get_model_params(self.model), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        set_model_params(self.model, parameters)
        _, loss, acc, bacc, _, _ = get_scores_sklearn(self.model, self.X_test, self.y_test)
        return loss, len(self.X_test), {"acc": acc, "bacc": bacc}

def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    seed = context.run_config["seed-random-state"]
    
    # Load data
    id_uci = context.run_config["id-uci"]
    X_train, X_test, y_train, y_test, classes = load_data(id_uci, partition_id, num_partitions, seed=seed)
    
    # Create model
    model_sklearn = context.run_config["model-sklearn"]
    local_epochs = context.run_config["local-epochs"]
    n_fearures = X_train.shape[1]
    model = get_model(model_sklearn, local_epochs, classes, n_fearures)
    
    return FlowerClient(partition_id, model, X_train, X_test, y_train, y_test).to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
