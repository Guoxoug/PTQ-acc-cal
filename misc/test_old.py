import torch
import torch.nn as nn
import torchvision as tv
import os
import json 
import pandas as pd
from models.model_generator import model_generator, load_weights_from_file
from torchsummary import summary
from utils.id_eval_utils import ECELoss, TopKError, print_results
from utils.ood_eval_utils import (
    ood_detect_results,
    rejection_ratio_results,
    uncertainties,
    get_ood_metrics_from_combined
)
from utils.data_utils import (
    Data,
    IMAGENET_TRANSFORMS,
    CIFAR_10_TRANSFORMS,
    CIFAR_100_TRANSFORMS, 
    DATASET_NAME_TRANSFORM_MAPPING,
    get_preprocessing_transforms
)

from utils.quant_utils import print_model, EditedFakeQuantize
import utils.quant_utils as quant_utils
from tqdm import tqdm
import copy
from torch.quantization import (
    disable_fake_quant, disable_observer, enable_fake_quant, FakeQuantize
)
from torch.quantization.observer import (
    HistogramObserver,
    MinMaxObserver,
    MovingAverageMinMaxObserver,
    MovingAveragePerChannelMinMaxObserver,
    PerChannelMinMaxObserver
)

from argparse import ArgumentParser

from utils.train_utils import get_filename

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--seed",
    default=0,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)
parser.add_argument(
    "--weights_path",
    type=str,
    default=None,
    help="Optional path to weights, if not automatically found from config."
)

args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set random seed
# prioritize config seed
if "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])
else:
    torch.manual_seed(args.seed)

# determinism in testing
torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = "16:8"

# set gpu
# bit of a hack to get around converting json syntax to bash
# deals with a list of integer ids
os.environ["CUDA_VISIBLE_DEVICES"] = str(
    config["gpu_id"]
).replace("[", "").replace("]", "")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


id_data = Data(
    **config["id_dataset"],
    test_only=True,
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"])
)

test_loader = id_data.test_loader

# get id dataset normalisation values
ood_data = [
    Data(
        **ood_config,
        transforms=get_preprocessing_transforms(
            ood_config["name"],
            id_dataset_name=config["id_dataset"]["name"]
        )
    )
    for ood_config in config["ood_datasets"]
]


# print transforms
print("="*80)
print(id_data.name)
print(id_data.test_set.transforms)
print("="*80)
for data in ood_data:
    print("="*80)
    print(data.name)
    print(data.test_set.dataset.transforms)
    print("="*80)


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

    logits, labels = get_logits_labels(model, id_data.test_loader, dev=dev)
    results = {}
    results["top1"] = top1(labels, logits)
    results["top5"] = top5(labels, logits)
    results["ece"] = ece(labels, logits)

    if ood_data is not None:
        ood_results = {}
        for data in ood_data:
            ood_logits, _ = get_logits_labels(model, data.test_loader, dev=dev)

            # balance the #samples between OOD and ID data
            ood_logits = ood_logits[:len(logits)]
            combined_logits = torch.cat([logits, ood_logits])
            # ID 0, OOD 1
            domain_labels = torch.cat(
                [torch.zeros(len(logits)), torch.ones(len(ood_logits))]
            )

            # gets different uncertainty metrics for combined ID and OOD
            metrics = uncertainties(combined_logits)
            res = ood_detect_results(
                domain_labels, metrics, mode="ROC"
            )

            res = {
                f"OOD {data.name} ROC " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

        results.update(ood_results)
    
    return results




# load floating point densenet model and evaluate
float_model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
)
# try and get weights 
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
        get_filename(config) + ".pth"
    )
try:
    load_weights_from_file(float_model, weights_path)
except:
    print("Failed to load weights, will be randomly initialised.")

# multigpu
model = torch.nn.DataParallel(float_model) if (
    config["data_parallel"] and torch.cuda.device_count() > 1
) else float_model

model.to(dev)

# some models may not support fusing modules (e.g. densenet)
try:
    float_model.fuse_model()
except:
    pass

# summary(float_model, (3,32,32))


# calibrate ptq model
quant_model = copy.deepcopy(float_model)

torch.backends.quantized.engine = 'fbgemm'

standard_model = copy.deepcopy(quant_model)
standard_model.qconfig = torch.quantization.QConfig(
    activation=MinMaxObserver.with_args(
        reduce_range=False,
        # quant_min=0,
        # quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine

    ),

    weight=PerChannelMinMaxObserver.with_args(
        quant_min=-32,
        quant_max=31,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric
    )
)
# quant_model.qconfig = torch.quantization.get_default_qconfig('qnnpack')

print(f"Quantization configuration:\n{standard_model.qconfig}")

torch.quantization.prepare(standard_model, inplace=True)
standard_model.eval()
# calibrate/observe for 20 batches of 256
with torch.no_grad():
    for i, (inputs, labels) in enumerate(tqdm(test_loader)):
        labels, inputs = labels.to("cuda"), inputs.to("cuda")
        outputs = standard_model(inputs)
        if i == 20:
            break


# freeze quantizer parameters
# quant_model.apply(torch.quantization.disable_observer)
# quant_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
standard_model.to("cpu")
torch.quantization.convert(standard_model, inplace=True)
print_model(standard_model)

sim_model = copy.deepcopy(quant_model)
sim_model.qconfig = torch.quantization.QConfig(
    activation=EditedFakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        reduce_range=False,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_channel_affine,
        ch_axis=1

    ),

    weight=EditedFakeQuantize.with_args(
        observer=PerChannelMinMaxObserver,
        quant_min=-8,
        quant_max=7,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,
        ch_axis=1
    )
)

# sim_model.qconfig = torch.quantization.get_default_qat_qconfig()

print(f"Quantization configuration:\n{sim_model.qconfig}")

# insert observers and replace modules with
# QAT versions that do simulated quantization
sim_mapping = torch.quantization.get_default_qat_module_mappings()
# sim_mapping[nn.BatchNorm2d] = quant_utils.BatchNorm2d
torch.quantization.prepare_qat(sim_model, inplace=True, mapping=sim_mapping)

print_model(sim_model)

# simulated quantization layer us there but it only observes
sim_model.apply(disable_fake_quant)
sim_model.eval()

# calibrate/observe for 20 batches of 256
with torch.no_grad():
    for i, (inputs, labels) in enumerate(tqdm(test_loader)):
        labels, inputs = labels.to("cuda"), inputs.to("cuda")
        outputs = sim_model(inputs)
        if i == 20:
            break

# simulate ptq
sim_model.apply(disable_observer)
sim_model.apply(enable_fake_quant)

observer_dict = {}
torch.quantization.get_observer_dict(sim_model, observer_dict)
# print(observer_dict)

# list some modules
# print_model(float_model)
# print("="*40)
# print("="*40)
# print_model(standard_model)
# print("="*40)
# print("="*40)
# print_model(sim_model)
# print("="*40)
# print("="*40)
# evauluate the floating model

# list of results dictionaries
result_rows = []

fp_results = evaluate(float_model, id_data, ood_data=ood_data)
print_results(fp_results)

result_rows.append(fp_results)


# evaluate pytorch standard model
sim_model.to("cuda")
sim_results = evaluate(sim_model, id_data, ood_data=ood_data, dev="cuda")
print_results(sim_results)

result_rows.append(sim_results)

# evaluate simulated model

# standard_results = evaluate(
#     standard_model, id_data, ood_data=ood_data, dev="cpu"
# )
# print_results(standard_results)

# result_rows.append(standard_results)


# results into DataFrame
result_df = pd.DataFrame(result_rows)
result_df["quant"] = ["none", "ptq"]

if config["results_save"]:
    spec = config["model"]["model_type"] + "_" + config["id_dataset"]["name"]
    savepath = os.path.join(config["results_savedir"], f"{spec}.csv")
    if not os.path.isfile('filename.csv'):
        result_df.to_csv(savepath, mode="a", header=True)
    else: # else it exists so append without writing the header
        result_df.to_csv(savepath, mode="a", header=False)

