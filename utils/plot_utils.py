import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cycler
from torch.nn.functional import softmax
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score
import seaborn as sns
import os
from utils.train_utils import get_filename
sns.set_theme()




def plot_ptq_err_swap_hist(
    delta_err_dict, 
    conf_dict,
    data_size,
    dataset_name,
    config, suffix="", num_bins=50
):
    """Plot histograms of swap confidence and change in error over bitwidth."""
    assert delta_err_dict.keys() == conf_dict.keys(), (
        "delta error and conf keys must match (should be in form afp, w4)"
    )
    spec = get_filename(config, seed=None)
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

    # suffix is there for custom filename
    filename = get_filename(config, seed=config["seed"]) + "_" \
        + dataset_name +\
        "_" + "err_conf_swaps" + suffix + ".pdf"
    path = os.path.join(save_dir, filename)

    # unpack dict into two lists
    labels, confs = zip(*conf_dict.items())
    _, err = zip(*delta_err_dict.items())
    err = np.array(err)

    # Computed quantities to aid plotting

    # y range
    hist_range = (0.0, 1.0)

    # x range
    # share the same range over all  histrograms
    binned_data_sets = [
        np.histogram(conf, range=hist_range, bins=num_bins)[0]
        for conf in confs
    ]

    binned_maximums = np.max(binned_data_sets, axis=1)

    x_locations = np.linspace(
        0,
        len(binned_maximums) * np.max(binned_maximums) * 0.825,
        len(binned_maximums)
    ) + np.max(binned_maximums) * 0.25
   

    # The bin_edges are the same for all of the histograms
    bin_edges = np.linspace(hist_range[0], hist_range[1], num_bins + 1)
    centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
    heights = np.diff(bin_edges)

    # Cycle through and plot each histogram
    fig, axes = plt.subplots(3,1, sharex=True, figsize=(6, 6))
    for x_loc, binned_data in zip(x_locations, binned_data_sets):
        lefts = x_loc - 0.5 * binned_data 
        axes[2].barh(
            centers, binned_data, height=heights, left=lefts,
            color="navy", alpha=.5
        )
    axes[2].set_xticks(x_locations)
    axes[2].set_xticklabels(labels)

    axes[2].set_ylabel("original confidence")
    axes[2].set_xlabel("precision")
    axes[2].set_ylim((
        -0.1,
        1
    ))

    # # add some text to legend
    axes[2].plot(
        [], [],linestyle="-", color="navy", alpha=.5, linewidth=2,
        label=f"distribution of swapped predictions"
    )
    axes[2].legend(loc="lower right")

    proportion_swapped = np.array([
        len(conf_dict[k])/data_size for k in conf_dict.keys()
    ])

    axes[0].plot(
        x_locations, proportion_swapped * 100,
        color="navy", linestyle="dashed", alpha=.7, marker="o",
        label="% swapped"
    )

    axes[0].plot(
        x_locations, err * 100, 
        color="indianred", linestyle="dashed", alpha=.7, marker="o",
        label="$\Delta$% error rate"
    )

    axes[0].set_ylabel("percentage")

    axes[0].legend()

    axes[1].plot(
        x_locations, err/proportion_swapped,
        color="black", linestyle="dashed", alpha=.7, marker="o",
        label="$\Delta$% error rate / % swapped"
    )

    axes[1].set_ylabel("ratio")
    axes[1].legend()
    
    # set limits for presentation

    axes[0].set_ylim((-5, 35))
    axes[1].set_ylim((-0.05, 0.4))
    axes[2].set_ylim((0,1))
    # minor gridlines need this extra code
    for ax in axes:
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='minor', color='w', linewidth=0.075)

        # need this so that shift is seen
        ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved to: {path}")


def plot_reliability_curve(
    logits, labels, legend_labels=[], file_path=None, n_bins=15, title=None,
    histogram=True, one_color=True
):
    """Plot a reliability curve given logits and labels.
    
    Also can optionally have a histogram of confidences.
    """

    # expect to iterate through a list
    if type(logits) == np.ndarray:
        logits = [logits]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    acc = np.array([
        accuracy_score(labels, data.argmax(axis=-1)) for data in logits 
    ]).mean() * 100

    ax.plot(
        [0, 1], [0, 1],
        linestyle='--', color="black", label="perfect calibration"
    )

    # put overall acc in title
    ax.set_title(f"overall accuracy: {acc:.2f}%")

    # color cycle is gradual
    n_plots = len(logits)
    color = plt.cm.magma(np.linspace(0, 0.8, n_plots))
    color[:, -1] = 0.5  # alpha
    new_cycler = cycler.cycler(color=color)
    ax.set_prop_cycle(new_cycler)
    overall_probs = []
    for i, data in enumerate(logits):
        if type(data) != np.ndarray:
            data = np.array(data)
        probs = softmax(torch.tensor(data), dim=-1).numpy()
        overall_probs.append(probs)
        fop, mpv = calibration_curve(
            (labels == np.argmax(probs, axis=-1)),
            np.max(probs, axis=-1),
            n_bins=n_bins,
            strategy="quantile"
        )
        if legend_labels:
            legend_label = legend_labels[i]
        else:
            legend_label = None
        if one_color:
            ax.plot(mpv, fop, color="indianred", alpha=0.5, label=legend_label)
        else:
            ax.plot(mpv, fop, label=legend_label)

    if one_color:
        ax.set_ylabel('accuracy', color="indianred")
    else:
        ax.set_ylabel('accuracy')
    ax.set_xlabel('confidence')

    # override acc for custom title
    if title is not None:
        ax.set_title(title)

    # histogram shows density of predictions wrt confidence
    if histogram:
        overall_probs = np.concatenate(overall_probs)
        confs = overall_probs.max(axis=-1)
        ax2 = ax.twinx()
        ax2.hist(
            confs,
            density=True,
            bins=20,
            alpha=0.2,
            color="navy",
            range=(0,1)
        )
        ax2.set_ylabel("density", color="navy")
        ax2.grid(False)
    ax.legend()
    fig.tight_layout()
    if file_path is not None:
        fig.savefig(file_path)
    else:
        plt.show()

