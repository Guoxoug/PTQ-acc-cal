from models.resnet import CifarResNet, ResNet
from models.mobilenet_v2 import MobileNetV2

import torch

MODEL_TYPES = [
    "resnet", "cifar_resnet", "mobilenetv2", 
    "resnet50", "cifar_resnet20", "cifar_resnet56" # specific configurations
]

MODEL_TYPE_MAPPINGS = {
    "resnet":ResNet,
    "resnet50": ResNet,
    "cifar_resnet": CifarResNet,
    "cifar_resnet20": CifarResNet,
    "cifar_resnet56": CifarResNet,
    "mobilenetv2": MobileNetV2,
}

def model_generator(model_type:str, **model_params) -> torch.nn.Module:
    """Construct a model following the supplied parameters."""
    assert model_type in MODEL_TYPES, (
        f"model type not supported"
        f"needs to be in {MODEL_TYPES}"    
    )

    # select model class
    Model = MODEL_TYPE_MAPPINGS[model_type]

    # override with proper values
    if model_type == "cifar_resnet20":
        model_params["layers"] = [3,3,3]
    if model_type == "cifar_resnet56":
        model_params["layers"] = [9, 9, 9]
    if model_type == "resnet50":
        model_params["layers"] = [3, 4, 6, 3]
        model_params["block"] = "bottleneck"
    if model_type == "mnasnet1":
        model_params["alpha"] = 1.0

    # generic unpacking of pararmeters, need to match config file with 
    # model definition
    model = Model(**model_params)
    return model

def load_weights_from_file(
    model, weights_path, dev="cuda", keep_last_layer=True
):
    """Load parameters from a path of a file of a state_dict."""
    state_dict = torch.load(weights_path, map_location=dev)
    if not keep_last_layer:

        # filter out final linear layer weights
        state_dict = {
            key: params for (key, params) in state_dict.items()
            if "classifier" not in key and "fc" not in key
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=True)


