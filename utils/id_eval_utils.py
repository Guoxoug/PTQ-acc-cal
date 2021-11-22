import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


# adapted from 
# https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15, percent=True):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.percent = percent

    def forward(self, labels, logits):
        softmaxes = logits.softmax(dim=-1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

        return 100* ece.item() if self.percent else ece.item()


class TopKError(nn.Module):
    """
    Calculate the top-k error rate of a model. 
    """
    def __init__(self, k=1, percent=True):
        super().__init__()
        self.k = k
        self.percent = percent 

    def forward(self, labels, outputs):
        # get rid of empty dimensions
        if type(labels) == np.ndarray:
            labels = torch.tensor(labels)
        if type(outputs) == np.ndarray:
            outputs = torch.tensor(outputs)
        labels, outputs = labels.squeeze(), outputs.squeeze()
        _, topk = outputs.topk(self.k, dim=-1)
        # same shape as topk with repeated values in class dim
        labels = labels.unsqueeze(-1).expand_as(topk)
        acc = torch.eq(labels, topk).float().sum(dim=-1).mean()
        err = 1 - acc
        err = 100 * err if self.percent else err
        return err.item()
        
# logit swapping --------------------------------------------------------------

def get_swaps_confidence(
    orig_logits: np.ndarray, quant_logits: np.ndarray
):
    """Given pre and post quantization logits, get the top logit swaps.
    
    Return an array that lists the confidence of each swap. 
    """

    # find samples where swaps occur
    max_indices = orig_logits.argmax(axis=-1)
    quant_max_indices = quant_logits.argmax(axis=-1)
    swap_indices = (max_indices != quant_max_indices).nonzero()

    # get confidences
    probs = F.softmax(torch.tensor(orig_logits), dim=-1).numpy()
    swapped_probs = probs[swap_indices]
    confs = swapped_probs.max(axis=-1)

    return confs

# printing --------------------------------------------------------------------

def print_results(results: dict):
    """Print the results in a results dictionary."""
    print("="*80)
    for k, v in results.items():
        if type(v) == float:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    print("="*80)

