import os
import time
import math
import csv
import yaml
import logging
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import segmentation_models_pytorch as smp

from utils import (
    set_seed, get_model, get_loss_function, get_optimizer,
    get_scheduler, pixel_accuracy, setup_logging)
from dataset import get_dataloaders

def _save_confusion_matrix(stats, path,
                           class_labels=('Predicted Road', 'Predicted Background'),
                           true_labels=('Actual Road', 'Actual Background')):
    tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
    total_tp, total_fp, total_fn, total_tn = tp.item(), fp.item(), fn.item(), tn.item()
    matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])

    group_counts = [f"{value:,.0f}" for value in matrix.flatten()]
    total_pixels = matrix.sum()
    group_percentages = [f"{value / total_pixels:.2%}" for value in matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues',
                xticklabels=list(class_labels),
                yticklabels=list(true_labels))
    plt.xlabel('Model Prediction')
    plt.ylabel('Actual Value')
    plt.title('Confusion Matrix for Best Epoch', fontsize=14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    logging.info(f"📈 Confusion matrix saved at: {path}")

def _save_training_plots(history, config):
    plot_dir = config['training']['plot_dir']
    num_epochs = len(history['train_loss'])
    os.makedirs(plot_dir, exist_ok=True)

    for key, value in history.items():
        np.save(os.path.join(plot_dir, f'{key}.npy'), np.array(value))
    logging.info(f"\nTraining history saved to '{plot_dir}' directory.")

    epochs_range = range(1, num_epochs + 1)
    plt.style.use('seaborn-v0_8-whitegrid')

    plot_pairs = {
        f'Loss (Main: {config["loss"]["name"]})': ('train_loss', 'val_loss'),
        'IoU Score': ('train_iou_score', 'val_iou_score'),
        'F1-Score': ('train_f1_score', 'val_f1_score'),
        'Pixel Accuracy': ('train_accuracy', 'val_accuracy'),
        'Dice Loss': ('train_dice_loss', 'val_dice_loss'),
        'Focal Loss': ('train_focal_loss', 'val_focal_loss'),
    }

    num_plots = len(plot_pairs)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
    fig.suptitle('Model Training Results', fontsize=16, y=0.97)
    axes = axes.flatten()

    for i, (title, keys) in enumerate(plot_pairs.items()):
        ax = axes[i]
        x = list(epochs_range)
        for series_key, label in [(keys[0], 'Train'), (keys[1], 'Validation')]:
            if series_key in history:
                ax.plot(x, history[series_key], 'o-', label=label)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()

    for i in range(len(plot_pairs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(plot_dir, "training_metrics_summary.png")
    plt.savefig(plot_path)
    logging.info(f"Summary plot saved at: {plot_path}")
    plt.close()

def _log_training_times(plot_dir, epoch_times, total_time):
    time_log_path = os.path.join(plot_dir, "training_times.txt")
    with open(time_log_path, 'w', encoding='utf-8') as f:
        f.write("TRAINING TIME REPORT\n")
        f.write("=" * 30 + "\n")
        for i, t in enumerate(epoch_times):
            f.write(f"Epoch {i + 1:02d} Time: {t:.2f} seconds\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total training time: {total_time / 60:.2f} minutes ({total_time:.2f} seconds)\n")
    logging.info(f"📄 Time report saved at: {time_log_path}")

def _suffix_from_config(config):
    model_name = config['model']['name']
    if model_name == "EfficientViT-Seg":
        model_zoo_name = config['model'].get('efficientvit_params', {}).get('model_zoo_name', 'default')
        model_variant = model_zoo_name.split('-')[-2]
        return f"{config['data']['dataset_name']}_{config['loss']['name']}_{model_name}_{model_variant}"
    else:
        encoder_name = config['model'].get('encoder_name', 'no_encoder')
        return f"{config['data']['dataset_name']}_{config['loss']['name']}_{model_name}_{encoder_name}"


def train_once(config: dict) -> dict:
    file_suffix = _suffix_from_config(config)
    config['training']['model_path'] = config['training'].get('model_path',
        f"model/model_{file_suffix}.pt")
    config['training']['plot_dir'] = config['training'].get('plot_dir',
        f"plot/plot_{file_suffix}")

    os.makedirs(config['training']['plot_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['training']['model_path']), exist_ok=True)

    set_seed(config.get('seed', 42))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    setup_logging(config)
    logging.info(f"Config suffix: {file_suffix}")
    logging.info(f"Device: {device}")

    csv_log_path = os.path.join(config['training']['plot_dir'], 'epoch_results.csv')
    csv_headers = [
        'timestamp', 'epoch', 'val_loss', 'val_iou', 'val_f1', 'val_accuracy',
        'val_dice_loss', 'val_focal_loss', 'train_loss', 'train_iou', 'train_f1',
        'train_accuracy', 'train_dice_loss', 'train_focal_loss', 'epoch_time_s'
    ]
    with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(csv_headers)

    logging.info("🔄 Loading data...")
    train_loader, val_loader = get_dataloaders(config)

    logging.info("🔧 Initializing model, loss, optimizer, scheduler...")
    model = get_model(config).to(device)
    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    loss_mode = "binary" if config['model']['classes'] == 1 else "multiclass"
    dice_loss_fn = smp.losses.DiceLoss(mode=loss_mode)
    focal_loss_fn = smp.losses.FocalLoss(mode=loss_mode)

    history = {
        "train_loss": [], "val_loss": [], "train_iou_score": [], "val_iou_score": [],
        "train_f1_score": [], "val_f1_score": [], "train_accuracy": [], "val_accuracy": [],
        "train_dice_loss": [], "val_dice_loss": [], "train_focal_loss": [], "val_focal_loss": []
    }

    best_val_iou = 0.0
    best_epoch_stats = {}
    start_time = time.time()
    epoch_times = []

    num_epochs = config['training']['num_epochs']
    for epoch in range(num_epochs):
        epoch_start = time.time()

        model.train()
        running_train_loss, train_dice_loss_sum, train_focal_loss_sum = 0.0, 0.0, 0.0
        total_train_tp, total_train_fp, total_train_fn, total_train_tn = 0, 0, 0, 0
        train_accuracy_sum = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train ({config['model'].get('encoder_name', '')})"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if config['model']['name'] == "EfficientViT-Seg" and outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            with torch.no_grad():
                train_dice_loss_sum += dice_loss_fn(outputs, masks).item()
                train_focal_loss_sum += focal_loss_fn(outputs, masks).item()

                if loss_mode == "multiclass":
                    preds = torch.argmax(outputs, dim=1)
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds.long(), masks.long(), mode=loss_mode,
                        num_classes=config['model']['classes']
                    )
                else:
                    preds = (torch.sigmoid(outputs) > 0.5)
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds.long(), masks.long(), mode=loss_mode
                    )

                total_train_tp += tp.sum()
                total_train_fp += fp.sum()
                total_train_fn += fn.sum()
                total_train_tn += tn.sum()
                train_accuracy_sum += pixel_accuracy(outputs, masks) * images.size(0)

        history['train_loss'].append(running_train_loss / len(train_loader))
        history['train_dice_loss'].append(train_dice_loss_sum / len(train_loader))
        history['train_focal_loss'].append(train_focal_loss_sum / len(train_loader))
        history['train_accuracy'].append(train_accuracy_sum / len(train_loader.dataset))
        history['train_iou_score'].append(
            smp.metrics.iou_score(total_train_tp, total_train_fp, total_train_fn, total_train_tn, reduction='micro').item()
        )
        history['train_f1_score'].append(
            smp.metrics.f1_score(total_train_tp, total_train_fp, total_train_fn, total_train_tn, reduction='micro').item()
        )

        model.eval()
        running_val_loss, val_dice_loss_sum, val_focal_loss_sum = 0.0, 0.0, 0.0
        total_val_tp, total_val_fp, total_val_fn, total_val_tn = 0, 0, 0, 0
        val_accuracy_sum = 0.0

        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val ({config['model'].get('encoder_name', '')})"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                if config['model']['name'] == "EfficientViT-Seg" and outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

                running_val_loss += loss_fn(outputs, masks).item()
                val_dice_loss_sum += dice_loss_fn(outputs, masks).item()
                val_focal_loss_sum += focal_loss_fn(outputs, masks).item()
                val_accuracy_sum += pixel_accuracy(outputs, masks) * images.size(0)

                if loss_mode == "multiclass":
                    preds = torch.argmax(outputs, dim=1)
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds.long(), masks.long(), mode=loss_mode,
                        num_classes=config['model']['classes']
                    )
                else:
                    preds = (torch.sigmoid(outputs) > 0.5)
                    tp, fp, fn, tn = smp.metrics.get_stats(
                        preds.long(), masks.long(), mode=loss_mode
                    )

                total_val_tp += tp.sum()
                total_val_fp += fp.sum()
                total_val_fn += fn.sum()
                total_val_tn += tn.sum()

        history['val_loss'].append(running_val_loss / len(val_loader))
        history['val_dice_loss'].append(val_dice_loss_sum / len(val_loader))
        history['val_focal_loss'].append(val_focal_loss_sum / len(val_loader))
        history['val_accuracy'].append(val_accuracy_sum / len(val_loader.dataset))

        current_val_iou = smp.metrics.iou_score(total_val_tp, total_val_fp, total_val_fn, total_val_tn, reduction='micro').item()
        history['val_iou_score'].append(current_val_iou)
        history['val_f1_score'].append(
            smp.metrics.f1_score(total_val_tp, total_val_fp, total_val_fn, total_val_tn, reduction='micro').item()
        )

        scheduler.step(history["val_loss"][-1])
        if current_val_iou > best_val_iou:
            best_val_iou = current_val_iou
            torch.save(model.state_dict(), config['training']['model_path'])
            best_epoch_stats = {'tp': total_val_tp, 'fp': total_val_fp, 'fn': total_val_fn, 'tn': total_val_tn}

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_data_row = [
            timestamp_now, epoch + 1,
            f"{history['val_loss'][-1]:.4f}", f"{history['val_iou_score'][-1]:.4f}",
            f"{history['val_f1_score'][-1]:.4f}", f"{history['val_accuracy'][-1]:.4f}",
            f"{history['val_dice_loss'][-1]:.4f}", f"{history['val_focal_loss'][-1]:.4f}",
            f"{history['train_loss'][-1]:.4f}", f"{history['train_iou_score'][-1]:.4f}",
            f"{history['train_f1_score'][-1]:.4f}", f"{history['train_accuracy'][-1]:.4f}",
            f"{history['train_dice_loss'][-1]:.4f}", f"{history['train_focal_loss'][-1]:.4f}",
            f"{epoch_time:.2f}"
        ]
        with open(csv_log_path, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(epoch_data_row)

        logging.info(
            f"Epoch {epoch + 1:02d}/{num_epochs} | "
            f"Val Loss: {history['val_loss'][-1]:.4f} | Val IoU: {history['val_iou_score'][-1]:.4f} | "
            f"🕒 {epoch_time:.2f}s"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_training_time = time.time() - start_time
    _log_training_times(config['training']['plot_dir'], epoch_times, total_training_time)
    _save_training_plots(history, config)

    if best_epoch_stats:
        cm_path = os.path.join(config['training']['plot_dir'], 'confusion_matrix.png')
        _save_confusion_matrix(best_epoch_stats, cm_path)

    summary = {
        "best_val_iou": best_val_iou,
        "model_path": config['training']['model_path'],
        "plot_dir": config['training']['plot_dir'],
        "csv_log_path": csv_log_path
    }
    logging.info(f"✅ Best model saved at: {summary['model_path']}")
    logging.info(f"📊 Per-epoch results saved at: {summary['csv_log_path']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--encoders', nargs='*', default=None,
                        help='Optional: list encoder names to run sequentially (e.g., mobileone_s0 mobileone_s1)')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)

    if not args.encoders:
        return train_once(base_config)

    results = []
    for enc in args.encoders:
        exp = yaml.safe_load(yaml.dump(base_config))
        exp['model']['encoder_name'] = enc
        print(f"\n==== Running experiment: {exp['model']['name']} | {enc} ====")
        try:
            res = train_once(exp)
            results.append((enc, res.get("best_val_iou", None)))
        except Exception as e:
            print(f"ERROR training with encoder {enc}: {e} (continue)")
            continue
    print("\n==== SUMMARY ====")
    for enc, iou in results:
        print(f"{enc}: best IoU = {iou:.4f}" if iou is not None else f"{enc}: failed")
    return results


if __name__ == '__main__':
    main()