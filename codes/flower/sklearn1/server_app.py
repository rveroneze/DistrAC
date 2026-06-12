"""sklearn1: A Flower / sklearn app."""

from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, NDArrays, Scalar, FitRes, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common.logger import log

import numpy as np
from logging import INFO
from typing import Dict, Optional, Tuple, Union
import logging

from sklearn1.task import get_model, get_model_params, set_model_params

# ###############################################################################
# Tells Python where your project root is for your current terminal session
# It allows imports to work without complex build files.
# On Linux/macOS, run the command:
# export PYTHONPATH=/home/veroneze/Documents/github/AIDE/codes
# => Replace /path/to/your/project/root with your actual path
from common.my_utils import get_data_uci, get_scores_sklearn
# ###############################################################################


class SaveModelStrategy(FedAvg):
    def __init__(self, context: Context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = context
    
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        
        # Get values from context
        save_each_rounds = self.context.run_config["save-each-rounds"]
        num_server_rounds = self.context.run_config["num-server-rounds"]
        id_uci = self.context.run_config["id-uci"]
        seed = self.context.run_config["seed-random-state"]
        
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None and (server_round%save_each_rounds==0 or server_round==num_server_rounds):
            # Convert `Parameters` to `list[np.ndarray]`
            aggregated_ndarrays: list[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
            
            # Save aggregated_ndarrays to disk
            # print(f"Saving round {server_round} aggregated_ndarrays...")
            alg = 'LR' if 'LogisticRegression' in self.context.run_config["model-sklearn"] else 'MLP'
            np.savez(f"{alg}_DATA{id_uci}_SEED{seed}_round-{server_round}-params.npz", *aggregated_ndarrays)
            
        return aggregated_parameters, aggregated_metrics


def get_evaluate_fn(model, X_test, y_test):
    """Return an evaluation function for server-side evaluation."""

    # The `evaluate` function will be called after every round
    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        set_model_params(model, parameters)  # Update model with the latest parameters
        y_pred, loss, acc, bacc, _, _ = get_scores_sklearn(model, X_test, y_test)
        log(INFO, "Test loss: %.4f", loss)
        log(INFO, "Test accuracy: %.4f", acc)
        log(INFO, "Test b. accuracy: %.4f", bacc)
        return loss, {"acc": acc, "bacc": bacc}

    return evaluate


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    model_sklearn = context.run_config["model-sklearn"]
    local_epochs = context.run_config["local-epochs"]
    id_uci = context.run_config["id-uci"]
    seed = context.run_config["seed-random-state"]
    
    _, _, X_test, _, y_test, classes = get_data_uci(id_uci, seed=seed)
    n_fearures = X_test.shape[1]
    model = get_model(model_sklearn, local_epochs, classes, n_fearures)
    
    # Get model initial parameters
    initial_parameters = ndarrays_to_parameters(get_model_params(model))
    
    # Define strategy
    strategy = SaveModelStrategy(
        context=context,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=initial_parameters,
        evaluate_fn=get_evaluate_fn(model, X_test, y_test),  # <-- pass the centralized evaluation function
    )
    config = ServerConfig(num_rounds=num_rounds)
    
    # Configure logging to file
    alg = 'LR' if 'LogisticRegression' in model_sklearn else 'MLP'
    logging.basicConfig(
        filename=f"{alg}_DATA{id_uci}_SEED{seed}.log",          # File to save logs
        filemode="w",                   # Overwrite each time (use "a" to append)
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO              # Or DEBUG, WARNING, etc.
    )
       
    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)