#!/bin/bash
cd /home/guoxia01/uncertainty-compression/src

# path the configuration file
config_path=experiment_configs/mobilenetv2_imagenet.json

# python train.py $config_path --seed 1
# python test.py $config_path --seed 1


echo "only testing"


python test.py $config_path --seed 0 --gpu 0 --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/mobilenet_v2-b0353104.pth
python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/mobilenetv2_imagenet/mobilenetv2_imagenet_0.csv --suffix single
python plot_additional_results.py $config_path --num_runs 1 --logits_path /work/guoxia01/experiment_results/mobilenetv2_imagenet/mobilenetv2_imagenet_0_logits.pth --suffix single
