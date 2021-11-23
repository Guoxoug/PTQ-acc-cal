#!/bin/bash
cd ..

# path the configuration file
config_path=experiment_configs/mobilenetv2_imagenet.json

# rather than using a generated weights file from training
# just use the pretrained on provided by pytorch
python test.py $config_path --seed 1 --gpu 0 --weights_path models/saved_models/mobilenetv2_imagenet/mobilenet_v2-b0353104.pth
python plot_results.py $config_path --num_runs 1
