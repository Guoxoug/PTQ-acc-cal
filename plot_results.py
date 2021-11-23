"""Take saved logits from test.py and plot reliability curves."""

from utils.plot_utils import (
    plot_ptq_err_swap_hist,
    plot_reliability_curve,
)
from utils.id_eval_utils import TopKError, get_swaps_confidence
from utils.train_utils import get_filename
import torch
import os
import json
import numpy as np
from argparse import ArgumentParser

from utils.data_utils import (
    Data,
    get_preprocessing_transforms
)


parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)
parser.add_argument(
    "--logits_path",
    type=str,
    default=None,
    help=(
        "directory where result logit files are kept,"
        "deduced from config by default"
    )
)

parser.add_argument(
    "--seeds",
    default=None,
    type=str,
    help="string containing random seeds, overrides default [1 to num_runs]."
)

parser.add_argument(
    "--num_runs",
    type=int,
    default=1,
    help="number of independent runs to average over"
)

parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="filename suffix to make file unique if needs be"
)
args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)


# list of seeds
# if we have multiple runs and no seeds assume they just start at 1
seeds = [i for i in range(1, args.num_runs + 1)] if (
    args.seeds is None
) else list(args.seeds)

# single plot
if args.logits_path is not None:
    logits_path = args.logits_path
    logits_paths = [logits_path]

# results path generated as results_savedir/arch_dataset
else:
    results_path = os.path.join(
        config["test_params"]["results_savedir"],
        get_filename(config, seed=None)
    )
    logits_paths = [
        os.path.join(
            results_path, get_filename(config, seed=seed) + "_logits.pth"
        )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth 
        for seed in seeds
    ]

# these are actually dictionaries 
# containing many difference quantization levels
print("Loading logits")
logits = [
    torch.load(path) for path in logits_paths
]

id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
    fast=False
)

# get test labels
labelled_datasets = [id_data]


labelled_dataset_names = [data.name for data in labelled_datasets]
labels = {
    data.name: np.array(data.test_set.targets)
    for data in
    labelled_datasets
}

print("plotting")

# reliability curves
for data in labelled_datasets:

    label = labels[data.name]
    


    # plot for a single precision over multiple runs
    for precision in logits[0]:
        logits_list = []
        for precision_logit_dict in logits:

            logits_list.append(
                precision_logit_dict[precision][data.name]
            )

        spec = get_filename(config, seed=None)
        save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
        filename = get_filename(config) + "_" \
            + precision.replace(", ", "") + "_" + \
            data.name + f"_reliability_{args.suffix}" + ".pdf"
        path = os.path.join(save_dir, filename)
        plot_reliability_curve(
            logits_list, label, file_path=path,
        )
        print(f"saved plot to: {path}")

  

# plotting swapping behaviour--------------------------------------------------

top1 = TopKError(k=1, percent=False)

# iterate over seeds
for j, precision_logit_dict in enumerate(logits):
    suffix = args.suffix
    swap_conf_dict = {}
    config["seed"] = seeds[j]

    precision_err_dict = {}

    # iterate over datasets
    for dataset_name in precision_logit_dict["afp, wfp"]:
        print(f"dataset: {dataset_name}")
        swap_conf_dict[dataset_name] = {}

        # get error dictionary only for labelled data
        if dataset_name in labelled_dataset_names:
            precision_err_dict[dataset_name] = {
                k: top1(
                    labels[dataset_name], 
                    precision_logit_dict[k][dataset_name]
                )
                for k in precision_logit_dict
            }
        

        # iterate over precisions
        for precision in precision_logit_dict:
            if precision != "afp, wfp":

                fp_logits = precision_logit_dict["afp, wfp"][dataset_name]
                quant_logits = precision_logit_dict[precision][dataset_name]
                fp_logits = fp_logits.numpy()
                quant_logits = quant_logits.numpy()

                # swaps is only for ID dataset
                if dataset_name in labelled_dataset_names:
                    swap_conf_dict[dataset_name][precision] = \
                        get_swaps_confidence(
                            fp_logits, quant_logits
                        )

    # only for labelled datasets
    delta_error_dict = {
        name:
        {
            k: precision_err_dict[name][k] \
                - precision_err_dict[name]["afp, wfp"]
            for k in precision_err_dict[name]
            if k != "afp, wfp"
        }
        for name in precision_err_dict
    }   



    
    for data in labelled_datasets:

        # uncomment below for MobileNetV2 readability
        # delta_error_dict[data.name].pop("a8, w5")
        # delta_error_dict[data.name].pop("a8, w4")
        # swap_conf_dict[data.name].pop("a8, w5")
        # swap_conf_dict[data.name].pop("a8, w4")
        
        plot_ptq_err_swap_hist(
            delta_error_dict[data.name],
            swap_conf_dict[data.name],
            len(data.test_set),
            data.name,
            config,
            suffix=suffix
        )

