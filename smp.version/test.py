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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import get_model, pixel_accuracy
from dataset import CustomDataset

MODEL_NAME_PART = [
    "DeepLabV3Plus", "Unet", "FPN", "DeepLabV3", "LinkNet",
    "UNet++", "DPT", "SegFormer", "MAnet", "PAN", "UPerNet"
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

    total_tp = tp.sum().item()
    total_fp = fp.sum().item()
    total_fn = fn.sum().item()
    total_tn = tn.sum().item()

    matrix = np.array([[total_tn, total_fp],
                       [total_fn, total_tp]])

    group_counts = [f"{value:,.0f}" for value in matrix.flatten()]
    total_pixels = matrix.sum()
    group_percentages = [f"{value / total_pixels:.2%}" for value in matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues',
                xticklabels=[f'Predicted "{class_names[0]}"', f'Predicted "{class_names[1]}"'],
                yticklabels=[f'Actual "{class_names[0]}"', f'Actual "{class_names[1]}"'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix on Test Set', fontsize=14)
    plt.savefig(path)
    plt.close()
    print(f"📈 Confusion matrix saved to: {path}")


def evaluate_model(config_path, model_path, data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Results will be saved to: {output_dir}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"🔧 Using device: {device}")

    print("🔄 Loading model...")
    model_name, encoder_name = parse_config_from_filename(model_path)
    if model_name and encoder_name:
        print(f"   -> Auto identify config from file cònig: {model_name} | {encoder_name} | {data_dir}")
        config['model']['name'] = model_name
        config['model']['encoder_name'] = encoder_name
    else:
        print(
            f"Not Auto identify config from file cònig: {config['model']['name']} | {config['model']['encoder_name']}")
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Successfully loaded model from: {model_path}")

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.ToTensor()

    test_image_paths = sorted(glob.glob(os.path.join(data_dir, "images/Test/*.png")))
    test_mask_paths = sorted(glob.glob(os.path.join(data_dir, "mask/Test/*.png")))

    assert len(
        test_image_paths) > 0, f"No images found in the test directory! Path: {os.path.join(data_dir, 'images/Test/*.png')}"
    assert len(test_image_paths) == len(test_mask_paths), "The number of images and masks in the test set do not match!"

    print(f"🔍 Found {len(test_image_paths)} images in the test set.")

    test_dataset = CustomDataset(
        image_paths=test_image_paths,
        mask_paths=test_mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
        config=config
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    total_pixel_accuracy = 0.0
    num_classes = config['model']['classes']
    loss_mode = "binary" if num_classes == 1 else "multiclass"

    print("\n--- Starting evaluation on the test set ---")
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            batch_accuracy = pixel_accuracy(outputs, masks)
            total_pixel_accuracy += batch_accuracy

            if num_classes == 1:
                pred_masks = (torch.sigmoid(outputs) > 0.5).long()
                tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks.long(), mode=loss_mode)
            else:
                pred_masks = torch.argmax(outputs, dim=1)
                tp, fp, fn, tn = smp.metrics.get_stats(pred_masks, masks.long(), mode=loss_mode,
                                                       num_classes=num_classes)

            total_tp += tp.sum()
            total_fp += fp.sum()
            total_fn += fn.sum()
            total_tn += tn.sum()

            # if i < 5:
            #     save_prediction_examples(images[0], masks[0], pred_masks[0], output_dir, i)

            for j in range(len(images)):
                global_index = i * test_loader.batch_size + j

                if global_index < 30:
                    save_prediction_examples(images[j], masks[j], pred_masks[j], output_dir, global_index)

    iou_score = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    f1_score = smp.metrics.f1_score(total_tp, total_fp, total_fn, total_tn, reduction='micro')
    final_pixel_accuracy = total_pixel_accuracy / len(test_loader)

    print("\n--- Evaluation Results ---")
    print(f"📊 Pixel Accuracy: {final_pixel_accuracy:.4f}")
    print(f"📊 IoU (Jaccard) Score: {iou_score.item():.4f}")
    print(f"📊 F1-Score: {f1_score.item():.4f}")
    print("--------------------------\n")

    results_path = os.path.join(output_dir, "test_results.txt")
    with open(results_path, 'w') as f:
        f.write("--- Evaluation Results ---\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {data_dir}\n")
        f.write("--------------------------\n")
        f.write(f"Pixel Accuracy: {final_pixel_accuracy:.4f}\n")
        f.write(f"IoU (Jaccard) Score: {iou_score.item():.4f}\n")
        f.write(f"F1-Score: {f1_score.item():.4f}\n")
    print(f"📝 Numerical results saved to: {results_path}")

    cm_path = os.path.join(output_dir, "test_confusion_matrix.png")
    test_stats = {'tp': total_tp, 'fp': total_fp, 'fn': total_fn, 'tn': total_tn}
    class_names = ["Background", "Object"]
    save_confusion_matrix(test_stats, cm_path, class_names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a Segmentation Model on the Test Set.")

    parser.add_argument('model_path', type=str,
                        help='Path to the trained model weights file (.pt).')

    parser.add_argument('data_dir', type=str,
                        help='Path to the root directory of the dataset.')

    parser.add_argument('--output-dir', type=str, default='test_results',
                        help='Directory to save evaluation results (default: test_results).')

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file (default: config.yaml).')

    args = parser.parse_args()

    evaluate_model(args.config, args.model_path, args.data_dir, args.output_dir)