#!/bin/bash
cd ..

# path the configuration file
config_path=experiment_configs/resnet50_imagenet.json

echo "only testing"

# rather than using a generated weights file from training
# just use the pretrained on provided by pytorch
python test.py $config_path --seed 0 --gpu 2 --weights_path models/saved_models/resnet50_imagenet/resnet50-19c8e357.pth
python plot_results.py $config_path --num_runs 1

