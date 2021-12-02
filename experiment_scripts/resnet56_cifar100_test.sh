#!/bin/bash

cd ..

# path the configuration file
config_path=experiment_configs/resnet56_cifar100.json

python test.py $config_path --gpu 0

python plot_results.py $config_path --num_runs 1

