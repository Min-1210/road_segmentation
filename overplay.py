import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import argparse
import random

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

def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_model(config):
    model_name = config['name']
    model_params = {k: v for k, v in config.items() if k != 'name'}
    if model_params.get('activation') == 'null':
        model_params['activation'] = None
    model_class = getattr(smp, model_name, None)
    if model_class:
        return model_class(**model_params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

def preprocess(image, device, size=(256, 256)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess(pred_mask, num_classes):
    if num_classes > 1:
        if pred_mask.ndim == 3:
            pred_mask = np.argmax(pred_mask, axis=0)
    return (pred_mask * 255).astype(np.uint8)

def load_model(model_config, model_path, device):
    print("🚀 Loading model...")
    model = get_model(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, model_config, image_path, device, save_path="output.png"):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image, device)
    with torch.no_grad():
        output = model(input_tensor)
        if model_config['classes'] == 1:
            pred = (torch.sigmoid(output) > 0.5).long()
        else:
            pred = torch.argmax(output, dim=1).long()
    pred_np = pred.squeeze().cpu().numpy()
    mask = postprocess(pred_np, model_config['classes'])
    Image.fromarray(mask).save(save_path)
    print(f"✅ Saved mask to {save_path}")

def predict_folder(model, model_config, folder_path, device, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    if not image_files:
        print(f"❌ No images found in folder: {folder_path}")
        return
    print(f"🔍 Found {len(image_files)} images. Starting prediction...")
    for fname in image_files:
        in_path = os.path.join(folder_path, fname)
        out_path = os.path.join(output_dir, fname)
        predict_image(model, model_config, in_path, device, save_path=out_path)
    print("✨ Prediction complete for all images in the folder.")

def main(args):
    model_config = {
        "name": "UNet++",
        "encoder_name": "mobileone_s4",
        "encoder_weights": "imagenet",
        "in_channels": 3,
        "classes": 2,
        "activation": None
    }

    parsed_model, parsed_encoder = parse_config_from_filename(args.model_path)
    if parsed_model and parsed_encoder:
        print("Auto write from Teminal:")
        print(f"   - Model: {parsed_model}")
        print(f"   - Encoder: {parsed_encoder}")
        model_config["name"] = parsed_model
        model_config["encoder_name"] = parsed_encoder
    else:
        print(
            f" Not identify!. Use default config : {model_config['name']} | {model_config['encoder_name']}")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Using device: {device}")

    model = load_model(model_config, args.model_path, device)
    print(f"Successfully loaded model from: {args.model_path}")

    if args.image_path:
        predict_image(model, model_config, args.image_path, device, save_path="overplay_result.png")
    elif args.folder_path:
        predict_folder(model, model_config, args.folder_path, device, output_dir="overplay_results")
    else:
        print("❌ Please provide either --image_path or --folder_path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Segmentation Inference Script with Auto-Config.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model weight file (.pt).")
    parser.add_argument("--image_path", type=str, default=None, help="Path to a single image for prediction.")
    parser.add_argument("--folder_path", type=str, default=None, help="Path to a folder of images for prediction.")

    args = parser.parse_args()
    main(args)

    # EX: python overplay.py --model_path "model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt" --image_path "test.jpg"
