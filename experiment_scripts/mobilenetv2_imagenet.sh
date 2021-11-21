#!/bin/bash
cd /home/guoxia01/uncertainty-compression/src

# path the configuration file
config_path=experiment_configs/mobilenetv2_imagenet.json

# python train.py $config_path --seed 1
# python test.py $config_path --seed 1

# if not already defined
if ! [ -z ${test_only} ]; then
  test_only=false 
fi
echo "only testing"
if ! [ $test_only ]; then
    for num in $(seq 1 1 5)
    do
        python train.py $config_path --seed $num
    done
fi

for num in $(seq 1 1 5)
do
    python test.py $config_path --seed $num
done
python process_results.py $config_path 5
python plot_results.py $config_path
python plot_additional_results.py $config_path --num_runs 5