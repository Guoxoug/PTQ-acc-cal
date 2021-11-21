#!/bin/bash
cd /home/guoxia01/uncertainty-compression/src

test_only=true

# cifar 
experiment_scripts/resnet20_cifar10.sh -t 1 &
experiment_scripts/resnet20_cifar100.sh -t 1 &
experiment_scripts/resnet56_cifar10.sh -t 1 &
experiment_scripts/resnet56_cifar100.sh -t 1 &

wait

# imagenet
experiment_scripts/mobilenetv2_imagenet_single.sh &
experiment_scripts/resnet50_imagenet_single.sh &

wait 
experiment_scripts/mnasnet1_imagenet_single.sh 



echo "finished testing"

