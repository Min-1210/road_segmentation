import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import yaml
import argparse
import glob
from tqdm import tqdm
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_model, pixel_accuracy
from dataset import CustomDataset

MODEL_NAME_PART = [
    "DeepLabV3Plus", "Unet", "FPN", "DeepLabV3", "LinkNet",
    "UNet++", "DPT", "SegFormer", "MAnet", "PAN", "UPerNet", "EfficientViT-Seg"
]

def parse_config_from_filename(filename):
    try:
        stem = os.path.basename(filename)
        stem = stem.replace("model_", "").replace(".pt", "")
        parts = stem.split('_')

        found_model = None
        model_part_index = -1

        for i, part in enumerate(parts):
            if part in MODEL_NAME_PART:
                found_model = part
                model_part_index = i
                break

        if not found_model:
            return None, None

        if found_model == "EfficientViT-Seg":
            return found_model, "efficientvit_encoder"
            
        encoder_parts = parts[model_part_index + 1:]
        found_encoder = '_'.join(encoder_parts)

        return found_model, found_encoder
    except Exception:
        return None, None

def save_prediction_examples(image_tensor, mask_tensor, pred_tensor, output_path, index):
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    image = inv_normalize(image_tensor).cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    mask = mask_tensor.cpu().squeeze().numpy()
    pred_mask = pred_tensor.cpu().squeeze().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample Prediction #{index}', fontsize=16)

    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(mask, cmap='gray')
    ax2.set_title("Ground Truth Mask")
    ax2.axis('off')

    ax3.imshow(pred_mask, cmap='gray')
    ax3.set_title("Predicted Mask")
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"prediction_sample_{index}.png"))
    plt.close()

def save_confusion_matrix(stats, path, class_names):
    tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
    matrix = np.array([[tp.sum().item(), fp.sum().item()],
                       [fn.sum().item(), tn.sum().item()]])

    total_pixels = matrix.sum()
    if total_pixels == 0: total_pixels = 1
    
    group_counts = [f"{value:,.0f}" for value in matrix.flatten()]
    group_percentages = [f"{value / total_pixels:.2%}" for value in matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues',
                xticklabels=[f'Pred {class_names[0]}', f'Pred {class_names[1]}'],
                yticklabels=[f'True {class_names[0]}', f'True {class_names[1]}'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set', fontsize=14)
    plt.savefig(path)
    plt.close()

def evaluate_model(config_path, model_path, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    model_name, encoder_name = parse_config_from_filename(model_path)
    if model_name:
        print(f"-> Auto-detected from filename: {model_name}")
        config['model']['name'] = model_name
        if model_name != "EfficientViT-Seg" and encoder_name:
             config['model']['encoder_name'] = encoder_name
    else:
        print(f"-> Using config file settings: {config['model']['name']}")
    
    model = get_model(config).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.ToTensor()

    test_image_paths = sorted(glob.glob(os.path.join(data_dir, "images/Test/*.png")))
    test_mask_paths = sorted(glob.glob(os.path.join(data_dir, "mask/Test/*.png")))
    
    if len(test_image_paths) == 0:
        print("Không tìm thấy ảnh test! Kiểm tra lại đường dẫn.")
        return

    test_dataset = CustomDataset(test_image_paths, test_mask_paths, image_transform, mask_transform, config)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_pixel_acc = 0.0
    num_classes = config['model']['classes']
    loss_mode = "binary" if num_classes == 1 else "multiclass"

    print("\n--- Starting evaluation ---")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader)):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

            total_pixel_acc += pixel_accuracy(outputs, masks)

            if num_classes == 1:
                pred_masks = (torch.sigmoid(outputs) > 0.5).long()
                tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks.long(), mode=loss_mode)
            else:
                pred_masks = torch.argmax(outputs, dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks.long(), mode=loss_mode, num_classes=num_classes)

            total_tp += tp.sum()
            total_fp += fp.sum()
            total_fn += fn.sum()
            total_tn += tn.sum()

            if i * test_loader.batch_size < 20:
                 for j in range(len(images)):
                    save_prediction_examples(images[j], masks[j], pred_masks[j], output_dir, i*test_loader.batch_size + j)

    iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    f1 = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    
    print(f"\nResult: Acc: {total_pixel_acc/len(test_loader):.4f} | IoU: {iou:.4f} | F1: {f1:.4f}")
    
    with open(os.path.join(output_dir, "test_results.txt"), 'w') as f:
        f.write(f"Model: {model_path}\nIoU: {iou:.4f}\nF1: {f1:.4f}\nAcc: {total_pixel_acc/len(test_loader):.4f}")
    
    save_confusion_matrix({'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn}, 
                          os.path.join(output_dir, "cm.png"), ["Background", "Road"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--output-dir', type=str, default='test_results')
    args = parser.parse_args()
    evaluate_model(args.config, args.model_path, args.data_dir, args.output_dir)