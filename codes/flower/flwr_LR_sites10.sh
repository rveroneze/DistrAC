#!/bin/bash
export PYTHONPATH=$(dirname "$PWD")

flwr run . --federation-config "options.num-supernodes=10" --run-config "num-server-rounds=17 model-sklearn='LogisticRegression()' id-uci=70 seed-random-state=138"
flwr run . --federation-config "options.num-supernodes=10" --run-config "num-server-rounds=14 model-sklearn='LogisticRegression()' id-uci=70 seed-random-state=148"
flwr run . --federation-config "options.num-supernodes=10" --run-config "num-server-rounds=29 model-sklearn='LogisticRegression()' id-uci=14 seed-random-state=138"
flwr run . --federation-config "options.num-supernodes=10" --run-config "num-server-rounds=26 model-sklearn='LogisticRegression()' id-uci=14 seed-random-state=148"
