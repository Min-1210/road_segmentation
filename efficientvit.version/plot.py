import os
import math
import logging
import numpy as np
import matplotlib.pyplot as plt


def save_training_plots(history, config):
    plot_dir = config['training']['plot_dir']
    os.makedirs(plot_dir, exist_ok=True)

    for key, value in history.items():
        np.save(os.path.join(plot_dir, f"{key}.npy"), np.array(value))
    logging.info("Training history saved to '%s'", plot_dir)

    if not history:
        logging.warning("History is empty, skip plotting.")
        return

    if "train_loss" in history:
        num_epochs = len(history["train_loss"])
    else:
        some_key = next(iter(history.keys()))
        num_epochs = len(history[some_key])

    epochs_range = range(1, num_epochs + 1)

    plt.style.use("seaborn-v0_8-whitegrid")

    plot_pairs = {
        f'Loss (Main: {config["loss"]["name"]})': ("train_loss", "val_loss"),
        "IoU Score": ("train_iou_score", "val_iou_score"),
        "F1-Score": ("train_f1_score", "val_f1_score"),
        "Pixel Accuracy": ("train_accuracy", "val_accuracy"),
        "Dice Loss": ("train_dice_loss", "val_dice_loss"),
        "Focal Loss": ("train_focal_loss", "val_focal_loss"),
    }

    num_plots = len(plot_pairs)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols,
                             figsize=(num_cols * 6, num_rows * 4.5))
    fig.suptitle("Model Training Results", fontsize=16, y=0.97)

    if not isinstance(axes, (list, np.ndarray)):
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, (train_key, val_key)) in enumerate(plot_pairs.items()):
        ax = axes[i]
        has_any = False

        if train_key in history:
            ax.plot(epochs_range, history[train_key], "o-", label="Train")
            has_any = True
        if val_key in history:
            ax.plot(epochs_range, history[val_key], "o-", label="Validation")
            has_any = True

        if has_any:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
        else:
            ax.set_visible(False)

    for j in range(len(plot_pairs), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(plot_dir, "training_metrics_summary.png")
    plt.savefig(plot_path)
    logging.info("Summary plot saved at: %s", plot_path)
    plt.close()
