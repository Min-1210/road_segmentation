import os
import time
import torch
import yaml
import segmentation_models_pytorch as smp
from tqdm import tqdm

from utils import set_seed, get_model, get_loss_function, get_optimizer, get_scheduler, pixel_accuracy
from dataset import get_dataloaders
from plot import save_training_plots

def main(config):
    dataset_name = config['data']['dataset_name']
    loss_name = config['loss']['name']
    model_name = config['model']['name']
    encoder_name = config['model']['encoder_name']
    file_suffix = f"{dataset_name}_{loss_name}_{model_name}_{encoder_name}"
    config['training']['model_path'] = f"model/model_{file_suffix}.pt"
    config['training']['plot_dir'] = f"plot/plot_{file_suffix}"

    set_seed(config['seed'])
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))

    os.makedirs(config['training']['plot_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['training']['model_path']), exist_ok=True)

    print("🔄 Tải dữ liệu...")
    train_loader, val_loader = get_dataloaders(config)

    print("🔧 Khởi tạo mô hình, hàm loss, và optimizer...")
    model = get_model(config).to(device)
    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    print(f"\n==================================================")
    print(f"Bắt đầu huấn luyện cho: {file_suffix}")
    print(f"==================================================")
    print(f"Mô hình: {config['model']['name']} ({config['model']['encoder_name']})")
    print(f"Hàm loss: {config['loss']['name']}")
    print(f"Số lớp: {config['model']['classes']}")
    print(f"Huấn luyện trên thiết bị: {device}")

    loss_mode = "binary" if config['model']['classes'] == 1 else "multiclass"
    metrics_to_track = {
        "iou_score": smp.metrics.iou_score,
        "f1_score": smp.metrics.f1_score,
        "accuracy": pixel_accuracy,
        "dice_loss": smp.losses.DiceLoss(mode=loss_mode),
        "jaccard_loss": smp.losses.JaccardLoss(mode=loss_mode),
        "focal_loss": smp.losses.FocalLoss(mode=loss_mode),
    }
    if config['model']['classes'] == 1:
        metrics_to_track["bce_loss"] = torch.nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "val_loss": []}
    for name in metrics_to_track:
        history[f"train_{name}"] = []
        history[f"val_{name}"] = []

    start_time = time.time()
    print("\n--- Bắt đầu quá trình huấn luyện ---")

    for epoch in range(config['training']['num_epochs']):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1} Train"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        history["train_loss"].append(running_train_loss / len(train_loader))

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                running_val_loss += loss.item()
        history["val_loss"].append(running_val_loss / len(val_loader))

        for phase in ['train', 'val']:
            loader = train_loader if phase == 'train' else val_loader
            total_tp, total_fp, total_fn, total_tn = None, None, None, None
            metric_sums = {name: 0.0 for name in metrics_to_track}
            with torch.no_grad():
                for images, masks in loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    if config['model']['classes'] == 1:
                        pred_masks = (torch.sigmoid(outputs) > 0.5).long()
                        tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks.long(), mode='binary')
                        tp, fp, fn, tn = tp.sum(), fp.sum(), fn.sum(), tn.sum()
                    else:
                        pred_masks = torch.argmax(outputs, dim=1)
                        tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks.long(), mode='multiclass', num_classes=config['model']['classes'])
                        tp, fp, fn, tn = tp.sum(dim=0), fp.sum(dim=0), fn.sum(dim=0), tn.sum(dim=0)
                    if total_tp is None:
                        total_tp, total_fp, total_fn, total_tn = tp, fp, fn, tn
                    else:
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
                        total_tn += tn
                    for name, func in metrics_to_track.items():
                        if name in ["iou_score", "f1_score"]:
                            score_tensor = func(total_tp, total_fp, total_fn, total_tn)
                            metric_sums[name] += score_tensor.mean().item() if score_tensor.numel() > 1 else score_tensor.item()
                        elif name == "accuracy":
                            metric_sums[name] += func(outputs, masks)
                        elif "loss" in name:
                            metric_sums[name] += func(outputs, masks).item()
            for name in metrics_to_track:
                history[f"{phase}_{name}"].append(metric_sums[name] / len(loader))
        scheduler.step(history["val_loss"][-1])
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1:02d}/{config['training']['num_epochs']} | Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | Val IoU: {history['val_iou_score'][-1]:.4f} | Val F1: {history['val_f1_score'][-1]:.4f} | 🕒 {epoch_time:.2f}s")

    end_time = time.time()
    print("\n--- Huấn luyện hoàn tất ---")
    print(f"⏱️ Tổng thời gian huấn luyện: {(end_time - start_time) / 60:.2f} phút")
    torch.save(model.state_dict(), config['training']['model_path'])
    print(f"✅ Mô hình đã được lưu tại: {config['training']['model_path']}")
    save_training_plots(history, config)
    print(f"==================================================\n")

if __name__ == '__main__':
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(config)
