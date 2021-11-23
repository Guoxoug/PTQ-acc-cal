import os
import torch
import json
import time
from utils.data_utils import (
    Data, get_preprocessing_transforms, TRAIN_DATASETS
)
from utils.id_eval_utils import ECELoss, TopKError
from utils.train_utils import (
    OPTIMIZER_MAPPING, 
    SCHEDULER_MAPPING, 
    AverageMeter, 
    ProgressMeter,
    get_filename,
    save_state_dict
)
from models.model_generator import model_generator


from argparse import ArgumentParser

# load arguments/parameters for script-----------------------------------------

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this training script"
)

parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)

parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu to override config to set the gpu id to use."
)

args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

assert config["id_dataset"]["name"] in TRAIN_DATASETS, "not valid train set"

# setup------------------------------------------------------------------------

# set random seed
# CL arg overrides value in config file
if args.seed is not None:
    torch.manual_seed(args.seed)

    # add seed into config dictionary
    config["seed"] = args.seed

elif "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])

# no seed in config or as CL arg
else:
    torch.manual_seed(0)
    config["seed"] = 0
print("using random seed: ", config["seed"])


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
print(f"using {dev} for training")
print(f"gpu: ", dev)



# load training dataset
# more training arguments are passed directly from the configuration json
training_data = Data(
    **config["id_dataset"],
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"])
)

# load the model
model = model_generator(
    config["model"]["model_type"],
    **config["model"]["model_params"]
).to(dev)


# directory to save weights from training
# weights will be loaded from here for testing
if not (
    "weights_path" in config["model"]
    and
    config["model"]["weights_path"] is not None
):
    # default directory 
    config["model"]["weights_path"] = os.path.join(
        "models/saved_models/",
        get_filename(config, seed=None)
    )

# make a directory if it doesn't already exist
if not os.path.exists(config["model"]["weights_path"]):
    os.mkdir(config["model"]["weights_path"])


# training loss
criterion = torch.nn.CrossEntropyLoss()

# optimizer and scheduler
# dictionaries map strings to optimizer and scheduler classes
optimizer = OPTIMIZER_MAPPING[config["train_params"]["optimizer"]](
    model.parameters(), **config["train_params"]["optimizer_params"]
)
scheduler = SCHEDULER_MAPPING[config["train_params"]["lr_scheduler"]](
    optimizer, **config["train_params"]["lr_scheduler_params"]
)


# train and eval functions ----------------------------------------------------

def train_epoch(train_loader, model, criterion, optimizer, epoch:int):
    """Train the model for one epoch of the dataset."""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')
    ece = AverageMeter("ECE", ":6.2f")

    ece_calc = ECELoss()
    top1_calc = TopKError(k=1)
    top5_calc = TopKError(k=5)
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5, ece],
        prefix=f"Epoch: [{epoch}]")

    # switch to train
    model.train()

    start = time.time()
    
    for i, (inputs, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - start)

        # move data to correct device
        inputs, targets = inputs.to(dev), targets.to(dev)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        # note that outputs should be logits
        # targets should be labels (no distillation)
        err1 = top1_calc(targets, outputs)
        err5 = top5_calc(targets, outputs)
        ece_res = ece_calc(targets, outputs)
        batch_size = inputs.size(0) # may be smaller for last batch of epoch
        losses.update(loss.item(), batch_size)
        top1.update(err1, batch_size)
        top5.update(err5, batch_size)
        ece.update(ece_res, batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - start)
        start = time.time()

        if i % 20 == 0:
            progress.display(i)


def evaluate_epoch(val_loader, model, criterion, epoch: int) -> dict:
    """Evaluate the model for one epoch of the validation dataset."""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Err@1', ':6.2f')
    top5 = AverageMeter('Err@5', ':6.2f')
    ece = AverageMeter("ECE", ":6.2f")

    ece_calc = ECELoss()
    top1_calc = TopKError(k=1)
    top5_calc = TopKError(k=5)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]")

    # switch to evaluation mode (e.g. freezes bn stats)
    model.eval()

    start = time.time()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(val_loader):

            # move data to correct device
            inputs, targets = inputs.to(dev), targets.to(dev)

            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            # note that outputs should be logits
            # targets should be labels (no distillation)
            err1 = top1_calc(targets, outputs)
            err5 = top5_calc(targets, outputs)
            ece_res = ece_calc(targets, outputs)
            batch_size = inputs.size(0) # may be smaller for last batch 
            losses.update(loss.item(), batch_size)
            top1.update(err1, batch_size)
            top5.update(err5, batch_size)
            ece.update(ece_res, batch_size)


            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i % 20 == 0:
                progress.display(i)
        
    print(
        (
            f"Err@1 {top1.avg:.2f} Err@5 {top5.avg:.2f} ECE {ece.avg:.2f} "
            f"Loss {losses.avg:.2f} "
        )
    )
    eval_res = {
        "err1": top1.avg, 
        "err5": top5.avg,
        "ece": ece.avg,
        "loss": losses.avg
    }

    return eval_res

        
# training loop----------------------------------------------------------------

for epoch in range(config["train_params"]["num_epochs"]):
    
    # train
    train_epoch(
        training_data.train_loader,
        model,
        criterion,
        optimizer,
        epoch
    )

    # reduce learning rate if at correct epoch
    scheduler.step()
    
    # evaluate
    if training_data.val_size > 0:
        res = evaluate_epoch(
            training_data.val_loader,
            model,
            criterion,
            epoch
        )

# save parameter values at the end of training for testing
save_state_dict(model, config=config, is_best=False)



    



