#!/bin/bash
cd /home/guoxia01/uncertainty-compression/src


# cifar 
for config_path in experiment_configs/resnet20_cifar10.json \
                experiment_configs/resnet20_cifar100.json \
                experiment_configs/resnet56_cifar10.json \
                experiment_configs/resnet56_cifar100.json
    do
        python plot_results.py $config_path
        python plot_additional_results.py $config_path --num_runs 1
    done

# imagenet

config_path=experiment_configs/resnet50_imagenet.json
# python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/resnet50_imagenet/resnet50_imagenet_42.csv --suffix single
python plot_additional_results.py $config_path --seeds 0 --num_runs 1 --logits_path /work/guoxia01/experiment_results/resnet50_imagenet/resnet50_imagenet_42_logits.pth --suffix single

config_path=experiment_configs/mnasnet1_imagenet.json
# python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/mnasnet1_imagenet/mnasnet1_imagenet_42.csv --suffix single
python plot_additional_results.py $config_path --seeds 0 --num_runs 1 --logits_path /work/guoxia01/experiment_results/mnasnet1_imagenet/mnasnet1_imagenet_42_logits.pth --suffix single

config_path=experiment_configs/mobilenetv2_imagenet.json
# python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/mobilenetv2_imagenet/mobilenetv2_imagenet_42.csv --suffix single
python plot_additional_results.py $config_path --seeds 0 --num_runs 1 --logits_path /work/guoxia01/experiment_results/mobilenetv2_imagenet/mobilenetv2_imagenet_42_logits.pth --suffix truncated

echo "finished plotting"

