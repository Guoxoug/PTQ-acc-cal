import torch
import torch.nn as nn
import os
import json 
import pandas as pd

from models.model_generator import model_generator, load_weights_from_file
from utils.id_eval_utils import (
    ECELoss, TopKError, print_results,
)

from utils.data_utils import (
    Data,
    get_preprocessing_transforms,
)

from utils.quant_utils import get_qconfig

from tqdm import tqdm
import copy
from torch.quantization import (
    disable_fake_quant, disable_observer, enable_fake_quant
)

from argparse import ArgumentParser

from utils.train_utils import get_filename

# argument parsing-------------------------------------------------------------
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
    help="Optional path to weights, overrides config."
)

parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="added to end of filenames to differentiate them if needs be"
)



args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set random seed
# prioritize arg seed
if args.seed is not None:
    torch.manual_seed(args.seed)
    # add seed into config dictionary
    config["seed"] = args.seed
elif "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])
else:
    torch.manual_seed(0)
    config["seed"] = 0


# determinism in testing
torch.backends.cudnn.benchmark = False

# set gpu
if args.gpu is not None:
    config["gpu_id"] = args.gpu
elif "gpu_id" in config and type(config["seed"]) == int:
    pass
else:
    config["gpu_id"] = 0

# set training device, defaults to cuda
dev = torch.device(
    "cuda:" + str(config["gpu_id"])
    if torch.cuda.is_available() 
    else "cpu"
)

print(f"using {dev} for testing")
print(f"gpu: ", dev)

# data-------------------------------------------------------------------------

id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
    fast=False
)

test_loader = id_data.test_loader
train_loader = id_data.train_loader # for ptq calibration

# print transforms
print("="*80)
print(id_data.name)
print(id_data.test_set.transforms)
print("="*80)

# helper functions ------------------------------------------------------------

def get_logits_labels(model, loader, dev="cuda"):
    """Get the model outputs for a dataloader."""
    model.eval()
    # get ID data
    label_list = []
    logit_list = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            labels, inputs = labels.to(dev), inputs.to(dev)
            outputs = model(inputs)
            label_list.append(labels.to("cpu"))
            logit_list.append(outputs.to("cpu"))
    logits, labels = torch.cat(logit_list, dim=0), torch.cat(label_list, dim=0)
    return logits, labels

def evaluate(model, id_data, ood_data=None, dev="cuda"):
    """Evaluate the model's topk error rate and ECE."""
    ece = ECELoss()
    top1 = TopKError(k=1, percent=True)
    top5 = TopKError(k=5, percent=True)
    nll = nn.CrossEntropyLoss()

    logits_dict = {}
    print(f"eval on: {id_data.name}")
    logits, labels = get_logits_labels(model, id_data.test_loader, dev=dev)

    # store logits for later
    logits_dict[f"{id_data.name}"] = logits.to("cpu")

    results = {}
    results["dataset"] = id_data.name
    results["top1"] = top1(labels, logits)
    results["top5"] = top5(labels, logits)
    results["nll"] = nll(logits, labels).item() # arguments are backwards
    results["ece"] = ece(labels, logits)
   
    return results, logits_dict


# evaluation-------------------------------------------------------------------

# load floating point model and evaluate
float_model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)

float_model.to(dev)

# try and get weights 
# prioritize CL args
if args.weights_path is not None:
    weights_path = args.weights_path
elif (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # where pretrained weights are
    weights_path = os.path.join(
        config["model"]["weights_path"],
        get_filename(config, seed=config["seed"]) + ".pth"
    )
else:
    raise IOError("no path given to load weights from")


print(f"Trying to load weights from: {weights_path}\n")
load_weights_from_file(float_model, weights_path, dev=dev)
print("Loading successful")

# list of results dictionaries
result_rows = []

# eval floating point model
fp_results, fp_logits = evaluate(float_model, id_data, dev=dev)

fp_results["seed"] = config["seed"]
fp_results["activations"] = "fp"
fp_results["weights"] = "fp"


print_results(fp_results)
result_rows.append(fp_results)


# BN layer fused into preceding convolution
# some models may not support fusing modules 
# fusing happens before module swapping
try:
    float_model.fuse_model()
    print("model fused")
except:
    print("couldn't fuse model")
    pass

# dictionary with precision as keys, containing logits from test data
# stored for later use
precision_logit_dict = {}
precision_logit_dict["afp, wfp"] = fp_logits

# iterate through quantization configurations and evaluate
for ptq_config in config["test_params"]["ptq_configs"]:

    sim_quant_model = copy.deepcopy(float_model)
   
    if "ptq_observer" in config["test_params"]:
        observer = config["test_params"]["ptq_observer"]
    else:
        observer = "minmax"

    # helper function that retreives the corresponding quantization config
    sim_quant_model.qconfig = get_qconfig(
        activations=ptq_config["activations"],
        weights=ptq_config["weights"],
        observer=observer
    )

    print(f"Quantization configuration:\n{sim_quant_model.qconfig}")

    # insert observers and replace modules with
    # QAT versions that do simulated quantization
    sim_mapping = torch.quantization.get_default_qat_module_mappings()
    
    # leave linear layers alone
    # i.e. final FC is fp
    try:
        if not sim_quant_model.q_last_layer and sim_quant_model.q:
            del sim_mapping[nn.Linear] 
    except:
        pass

    # fusing has already occurred 
    # so if it has happened BNs will all be already gone
    torch.quantization.prepare_qat(
        sim_quant_model, inplace=True, mapping=sim_mapping
    )

    # histogram observer has a buffer that needs to be sent to dev
    sim_quant_model.to(dev)

    # simulated quantization layer is there but it only observes
    sim_quant_model.apply(disable_fake_quant)
    sim_quant_model.eval()

    # calibrate/observe for 16 batches of 64 of training data
    print(80*"=")
    print("Calibrating")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(train_loader)):
            labels, inputs = labels.to(dev), inputs.to(dev)
            outputs = sim_quant_model(inputs)
            if i == 16 * 64//config["id_dataset"]["batch_size"]:
                break

    # simulate ptq
    sim_quant_model.apply(disable_observer)
    sim_quant_model.apply(enable_fake_quant)

    sim_quant_model.to(dev)
    sim_results, sim_logits = evaluate(
        sim_quant_model, id_data, dev=dev
    )
    sim_results["seed"] = config["seed"]

    # ptq_config is a dictionary with weight and activation bitwidths
    sim_results.update(ptq_config)
    print_results(sim_results)

    result_rows.append(sim_results)

    # add logits to dictionary 
    if config["test_params"]["logits_save"]:
        key = "a" + str(ptq_config["activations"]) + ", " \
            + "w" + str(ptq_config["weights"])
        precision_logit_dict[key] = sim_logits

# results into DataFrame
result_df = pd.DataFrame(result_rows)

# save to subfolder with dataset and architecture in name
# filename will have seed 
if config["test_params"]["results_save"]:
    spec = get_filename(config, seed=None)
    filename = get_filename(config, seed=config["seed"])
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    savepath = os.path.join(save_dir, f"{filename}{args.suffix}.csv")

    # just overwrite what's there
    result_df.to_csv(savepath, mode="w", header=True)

# save the logits from all precisions
if config["test_params"]["logits_save"]:
    spec = get_filename(config, seed=None)
    filename = get_filename(config, seed=config["seed"])
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    savepath = os.path.join(save_dir, f"{filename}_logits{args.suffix}.pth")
    torch.save(precision_logit_dict, savepath)

    

