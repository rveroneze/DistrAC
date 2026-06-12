#!/bin/bash
export PYTHONPATH=$(dirname "$PWD")

flwr run . --federation-config "options.num-supernodes=10" --run-config "num-server-rounds=195 model-sklearn='MLPClassifier(batch_size=22,hidden_layer_sizes=(41,20),max_iter=5000,random_state=42)' id-uci=14 seed-random-state=138"
flwr run . --federation-config "options.num-supernodes=10" --run-config "num-server-rounds=137 model-sklearn='MLPClassifier(batch_size=22,hidden_layer_sizes=(62,31),max_iter=5000,random_state=42)' id-uci=14 seed-random-state=148"
