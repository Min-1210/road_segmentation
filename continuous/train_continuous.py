import os
import time
import torch
import yaml
import logging
import segmentation_models_pytorch as smp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
from datetime import datetime

from utils import set_seed, get_model, get_loss_function, get_optimizer, get_scheduler, pixel_accuracy, setup_logging
from dataset import get_dataloaders

def log_training_times(plot_dir, epoch_times, total_time):
    time_log_path = os.path.join(plot_dir, "training_times.txt")
    with open(time_log_path, 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO THỜI GIAN HUẤN LUYỆN\n")
        f.write("=" * 30 + "\n")
        for i, t in enumerate(epoch_times):
            f.write(f"Thời gian Epoch {i + 1:02d}: {t:.2f} giây\n")
        f.write("=" * 30 + "\n")
        f.write(f"Tổng thời gian huấn luyện: {total_time / 60:.2f} phút ({total_time:.2f} giây)\n")
    logging.info(f"📄 Báo cáo thời gian đã được lưu tại: {time_log_path}")


def save_training_plots(history, config):
    plot_dir = config['training']['plot_dir']
    num_epochs = len(history['train_loss'])
    os.makedirs(plot_dir, exist_ok=True)

    for key, value in history.items():
        np.save(os.path.join(plot_dir, f'{key}.npy'), np.array(value))
    logging.info(f"\nLịch sử huấn luyện đã được lưu vào thư mục '{plot_dir}'")

    epochs_range = range(1, num_epochs + 1)
    plt.style.use('seaborn-v0_8-whitegrid')

    plot_pairs = {
        f'Loss (chính: {config["loss"]["name"]})': ('train_loss', 'val_loss'),
        'IoU Score': ('train_iou_score', 'val_iou_score'),
        'F1-Score': ('train_f1_score', 'val_f1_score'),
        'Pixel Accuracy': ('train_accuracy', 'val_accuracy'),
    }

    available_plots = {title: keys for title, keys in plot_pairs.items() if
                       (isinstance(keys, tuple) and keys[0] in history) or (isinstance(keys, str) and keys in history)}
    num_plots = len(available_plots)
    num_cols = 2
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows * 5))
    fig.suptitle('Kết quả Huấn luyện Mô hình', fontsize=16, y=0.97)
    axes = axes.flatten()

    for i, (title, keys) in enumerate(available_plots.items()):
        ax = axes[i]
        ax.plot(epochs_range, history[keys[0]], 'o-', label=f'Train')
        ax.plot(epochs_range, history[keys[1]], 'o-', label=f'Validation')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Giá trị')
        ax.legend()

    for i in range(len(available_plots), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(plot_dir, "training_metrics_summary.png")
    plt.savefig(plot_path)
    logging.info(f"Biểu đồ tổng hợp đã được lưu tại: {plot_path}")
    plt.close()


def main(config):
    file_suffix = f"{config['data']['dataset_name']}_{config['loss']['name']}_{config['model']['name']}_{config['model']['encoder_name']}"
    config['training']['model_path'] = f"model/model_{file_suffix}.pt"
    config['training']['plot_dir'] = f"plot/plot_{file_suffix}"

    set_seed(config.get('seed', 42))
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    os.makedirs(config['training']['plot_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['training']['model_path']), exist_ok=True)

    setup_logging(config)

    csv_log_path = os.path.join(config['training']['plot_dir'], 'epoch_results.csv')
    csv_headers = ['timestamp', 'epoch', 'val_loss', 'val_iou', 'val_f1', 'val_accuracy', 'train_loss', 'train_iou',
                   'train_f1', 'train_accuracy', 'epoch_time_s']
    with open(csv_log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    logging.info(f"\n{' BẮT ĐẦU THÍ NGHIỆM MỚI '.center(80, '=')}")
    logging.info(f"Cấu hình: {file_suffix}")
    logging.info(f"{'='.center(80, '=')}")

    logging.info("🔄 Tải dữ liệu...")
    train_loader, val_loader = get_dataloaders(config)

    logging.info("🔧 Khởi tạo mô hình, hàm loss, và optimizer...")
    model = get_model(config).to(device)
    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    loss_mode = "binary" if config['model']['classes'] == 1 else "multiclass"
    history = {"train_loss": [], "val_loss": [], "train_iou_score": [], "val_iou_score": [], "train_f1_score": [],
               "val_f1_score": [], "train_accuracy": [], "val_accuracy": []}
    best_val_iou = 0.0
    start_time = time.time()
    epoch_times = []

    for epoch in range(config['training']['num_epochs']):
        epoch_start = time.time()

        model.train()
        running_train_loss = 0.0
        total_train_tp, total_train_fp, total_train_fn = 0, 0, 0
        train_accuracy_sum = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train ({config['model']['encoder_name']})"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1) if loss_mode == "multiclass" else (torch.sigmoid(outputs) > 0.5)
                tp, fp, fn, _ = smp.metrics.get_stats(preds.long(), masks.long(), mode=loss_mode,
                                                      num_classes=config['model'][
                                                          'classes'] if loss_mode == "multiclass" else None)
                total_train_tp += tp.sum()
                total_train_fp += fp.sum()
                total_train_fn += fn.sum()
                train_accuracy_sum += pixel_accuracy(outputs, masks) * images.size(0)

        history['train_loss'].append(running_train_loss / len(train_loader))
        history['train_accuracy'].append(train_accuracy_sum / len(train_loader.dataset))
        history['train_iou_score'].append(
            smp.metrics.iou_score(total_train_tp, total_train_fp, total_train_fn, 0).item())
        history['train_f1_score'].append(smp.metrics.f1_score(total_train_tp, total_train_fp, total_train_fn, 0).item())

        model.eval()
        running_val_loss = 0.0
        total_val_tp, total_val_fp, total_val_fn = 0, 0, 0
        val_accuracy_sum = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val ({config['model']['encoder_name']})"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                running_val_loss += loss_fn(outputs, masks).item()
                val_accuracy_sum += pixel_accuracy(outputs, masks) * images.size(0)
                preds = torch.argmax(outputs, dim=1) if loss_mode == "multiclass" else (torch.sigmoid(outputs) > 0.5)
                tp, fp, fn, _ = smp.metrics.get_stats(preds.long(), masks.long(), mode=loss_mode,
                                                      num_classes=config['model'][
                                                          'classes'] if loss_mode == "multiclass" else None)
                total_val_tp += tp.sum()
                total_val_fp += fp.sum()
                total_val_fn += fn.sum()

        history['val_loss'].append(running_val_loss / len(val_loader))
        history['val_accuracy'].append(val_accuracy_sum / len(val_loader.dataset))
        current_val_iou = smp.metrics.iou_score(total_val_tp, total_val_fp, total_val_fn, 0).item()
        history['val_iou_score'].append(current_val_iou)
        history['val_f1_score'].append(smp.metrics.f1_score(total_val_tp, total_val_fp, total_val_fn, 0).item())
        scheduler.step(history["val_loss"][-1])

        if current_val_iou > best_val_iou:
            best_val_iou = current_val_iou
            torch.save(model.state_dict(), config['training']['model_path'])

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_data_row = [timestamp_now, epoch + 1, f"{history['val_loss'][-1]:.4f}",
                          f"{history['val_iou_score'][-1]:.4f}", f"{history['val_f1_score'][-1]:.4f}",
                          f"{history['val_accuracy'][-1]:.4f}", f"{history['train_loss'][-1]:.4f}",
                          f"{history['train_iou_score'][-1]:.4f}", f"{history['train_f1_score'][-1]:.4f}",
                          f"{history['train_accuracy'][-1]:.4f}", f"{epoch_time:.2f}"]
        with open(csv_log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(epoch_data_row)

        logging.info(
            f"Epoch {epoch + 1:02d} | Val Loss: {history['val_loss'][-1]:.4f} | Val IoU: {history['val_iou_score'][-1]:.4f} | 🕒 {epoch_time:.2f}s")

    end_time = time.time()
    total_training_time = end_time - start_time
    logging.info(f"\n--- Huấn luyện cho {file_suffix} hoàn tất ---")
    log_training_times(config['training']['plot_dir'], epoch_times, total_training_time)

    logging.info(f"✅ Mô hình tốt nhất đã được lưu tại: {config['training']['model_path']}")
    logging.info(f"📊 Kết quả từng epoch đã được lưu tại: {csv_log_path}")

    save_training_plots(history, config)
    logging.info(f"\n{' KẾT THÚC THÍ NGHIỆM '.center(80, '=')}\n")


if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)
