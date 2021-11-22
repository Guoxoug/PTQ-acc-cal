#!/bin/bash

cd ..

# path the configuration file
config_path=experiment_configs/resnet56_cifar100.json

python train.py $config_path --seed 1 --gpu 0
