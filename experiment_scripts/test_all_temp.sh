#!/bin/bash
cd /home/guoxia01/uncertainty-compression/src

suffix=""
# # cifar 
# # for config_path in experiment_configs/resnet20_cifar10.json \
# #                 experiment_configs/resnet20_cifar100.json \
# #                 experiment_configs/resnet56_cifar10.json \
# #                 experiment_configs/resnet56_cifar100.json

# python test.py experiment_configs/resnet20_cifar10.json \
#     --seed 1 --gpu 0 \
#     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/resnet20_cifar10/cifar_resnet20_cifar10_1.pth \
#     --suffix $suffix

# python plot_results.py experiment_configs/resnet20_cifar10.json \
#     --csv_path /work/guoxia01/experiment_results/cifar_resnet20_cifar10/cifar_resnet20_cifar10_1${suffix}.csv \
    # --suffix $suffix


# python plot_additional_results.py experiment_configs/resnet20_cifar10.json \
#     --seeds 1 --num_runs 1 \
#     --logits_path /work/guoxia01/experiment_results/cifar_resnet20_cifar10/cifar_resnet20_cifar10_1_logits${suffix}.pth \
# #     --suffix $suffix

# # python test.py experiment_configs/resnet20_cifar100.json \
# #     --seed 1 --gpu 0 \
# #     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/resnet20_cifar100/cifar_resnet20_cifar100_1.pth \
# #     --suffix $suffix

# python plot_results.py experiment_configs/resnet20_cifar100.json \
#     --csv_path /work/guoxia01/experiment_results/cifar_resnet20_cifar100/cifar_resnet20_cifar100_1${suffix}.csv \
#     # --suffix $suffix

# python plot_additional_results.py experiment_configs/resnet20_cifar100.json \
#     --seeds 1 --num_runs 1 \
#     --logits_path /work/guoxia01/experiment_results/cifar_resnet20_cifar100/cifar_resnet20_cifar100_1_logits${suffix}.pth \
# #     --suffix $suffix

# # python test.py experiment_configs/resnet56_cifar10.json \
# #     --seed 1 --gpu 0 \
# #     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/resnet56_cifar10/cifar_resnet56_cifar10_1.pth \
# #     --suffix $suffix

# python plot_results.py experiment_configs/resnet56_cifar10.json \
#     --csv_path /work/guoxia01/experiment_results/cifar_resnet56_cifar10/cifar_resnet56_cifar10_1${suffix}.csv \
#     # --suffix $suffix

# python plot_additional_results.py experiment_configs/resnet56_cifar10.json \
#     --seeds 1 --num_runs 1 \
#     --logits_path /work/guoxia01/experiment_results/cifar_resnet56_cifar10/cifar_resnet56_cifar10_1_logits${suffix}.pth \
# #     --suffix $suffix

# # python test.py experiment_configs/resnet56_cifar100.json \
# #     --seed 1 --gpu 0 \
# #     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/resnet56_cifar100/cifar_resnet56_cifar100_1.pth \
# #     --suffix $suffix


# python plot_results.py experiment_configs/resnet56_cifar100.json \
#     --csv_path /work/guoxia01/experiment_results/cifar_resnet56_cifar100/cifar_resnet56_cifar100_1${suffix}.csv \
#     # --suffix $suffix


# python plot_additional_results.py experiment_configs/resnet56_cifar100.json \
#     --seeds 1 --num_runs 1 \
#     --logits_path /work/guoxia01/experiment_results/cifar_resnet56_cifar100/cifar_resnet56_cifar100_1_logits${suffix}.pth \
# #     --suffix $suffix

# # imagenet

# python test.py experiment_configs/resnet50_imagenet.json \
#     --seed 0 --gpu 0 \
#     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/resnet50-0676ba61.pth \
#     --suffix $suffix &

# # config_path=experiment_configs/mnasnet1_imagenet.json
# # python test.py $config_path --seed 0 --gpu 1 --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/mnasnet1.0_top1_73.512-f206786ef8.pth --suffix $suffix &
# python test.py experiment_configs/mnasnet1_imagenet.json\
#     --seed 0 --gpu 0 \
#     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/mnasnet1.0_top1_73.512-f206786ef8.pth  \
#     --suffix $suffix 

# python test.py experiment_configs/mobilenetv2_imagenet.json \
#     --seed 0 --gpu 2 \
#     --weights_path /home/guoxia01/uncertainty-compression/src/models/saved_models/mobilenet_v2-b0353104.pth \
#     --suffix $suffix &



config_path=experiment_configs/resnet50_imagenet.json
python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/resnet50_imagenet/resnet50_imagenet_0${suffix}.csv # --suffix # $suffix
python plot_additional_results.py $config_path --seeds 0 --num_runs 1 --logits_path /work/guoxia01/experiment_results/resnet50_imagenet/resnet50_imagenet_0_logits${suffix}.pth # --suffix $suffix

 

config_path=experiment_configs/mnasnet1_imagenet.json
python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/mnasnet1_imagenet/mnasnet1_imagenet_0${suffix}.csv # --suffix $suffix 
python plot_additional_results.py $config_path --seeds 0 --num_runs 1 --logits_path /work/guoxia01/experiment_results/mnasnet1_imagenet/mnasnet1_imagenet_0_logits${suffix}.pth # --suffix $suffix

config_path=experiment_configs/mobilenetv2_imagenet.json
python plot_results.py $config_path --csv_path /work/guoxia01/experiment_results/mobilenetv2_imagenet/mobilenetv2_imagenet_0${suffix}.csv # --suffix $suffix
python plot_additional_results.py $config_path --seeds 0 --num_runs 1 --logits_path /work/guoxia01/experiment_results/mobilenetv2_imagenet/mobilenetv2_imagenet_0_logits${suffix}.pth # --suffix $suffix

echo "finished plotting"
