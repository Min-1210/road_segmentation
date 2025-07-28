import os
import time
import torch
import yaml
import segmentation_models_pytorch as smp
from tqdm import tqdm

from utils import set_seed, get_model, get_loss_function, get_optimizer, get_scheduler, pixel_accuracy
from dataset import get_dataloaders
from plot import save_training_plots


def main(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    os.makedirs(config['training']['plot_dir'], exist_ok=True)

    print("🔄 Tải dữ liệu...")
    train_loader, val_loader = get_dataloaders(config)

    print("🔧 Khởi tạo mô hình, hàm loss, và optimizer...")
    model = get_model(config).to(device)
    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)

    print(f"Mô hình: {config['model']['name']} ({config['model']['encoder_name']})")
    print(f"Hàm loss: {config['loss']['name']}")
    print(f"Số lớp: {config['model']['classes']}")
    print(f"Huấn luyện trên thiết bị: {device}")

    metrics_to_track = {
        "iou_score": smp.metrics.iou_score,
        "f1_score": smp.metrics.f1_score,
        "dice_loss": smp.losses.DiceLoss(mode="binary"),
        "jaccard_loss": smp.losses.JaccardLoss(mode="binary"),
        "bce_loss": torch.nn.BCEWithLogitsLoss(),
        "focal_loss": smp.losses.FocalLoss(mode="binary")
    }

    history = {"train_loss": [], "val_loss": []}
    metric_names = ["accuracy"] + list(metrics_to_track.keys())
    for name in metric_names:
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
        train_epoch_metrics = {name: 0.0 for name in metric_names}
        val_epoch_metrics = {name: 0.0 for name in metric_names}

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                running_val_loss += loss.item()
            history["val_loss"].append(running_val_loss / len(val_loader))

            num_classes = config['model']['classes']

            for loader_name, loader, epoch_metrics in [('train', train_loader, train_epoch_metrics),
                                                       ('val', val_loader, val_epoch_metrics)]:
                for images, masks in loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)

                    if num_classes == 1:
                        tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks.long(), mode='binary')
                    else:
                        tp, fp, fn, tn = smp.metrics.get_stats(outputs, masks.long(), mode='multiclass',
                                                               num_classes=num_classes)

                    epoch_metrics["accuracy"] += pixel_accuracy(outputs, masks)
                    for name, func in metrics_to_track.items():
                        if "loss" in name:
                            # smp losses work on predictions, not raw outputs
                            pred_for_loss = torch.argmax(outputs, dim=1) if num_classes > 1 else outputs
                            epoch_metrics[name] += func(pred_for_loss, masks).item()
                        else:
                            epoch_metrics[name] += func(tp, fp, fn, tn).sum().item()

        for name in metric_names:
            history[f"train_{name}"].append(train_epoch_metrics[name] / len(train_loader))
            history[f"val_{name}"].append(val_epoch_metrics[name] / len(val_loader))

        scheduler.step(history["val_loss"][-1])

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1:02d}/{config['training']['num_epochs']} | "
              f"Train Loss: {history['train_loss'][-1]:.4f} | Val Loss: {history['val_loss'][-1]:.4f} | "
              f"Val IoU: {history['val_iou_score'][-1]:.4f} | Val F1: {history['val_f1_score'][-1]:.4f} | "
              f"🕒 {epoch_time:.2f}s")

    end_time = time.time()
    print(f"\n--- Huấn luyện hoàn tất ---")
    print(f"⏱️ Tổng thời gian huấn luyện: {(end_time - start_time) / 60:.2f} phút")

    model_path = config['training']['model_path']
    torch.save(model.state_dict(), model_path)
    print(f"✅ Mô hình đã được lưu tại: {model_path}")

    save_training_plots(history, config)


if __name__ == '__main__':
    main()