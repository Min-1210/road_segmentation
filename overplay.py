import os
import torch
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as T
import segmentation_models_pytorch as smp
import argparse
import random

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
    model_name = config['model']['name']
    model_params = {k: v for k, v in config['model'].items() if k != 'name'}

    if model_params.get('activation') == 'null':
        model_params['activation'] = None

    model_map = {
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "Unet": smp.Unet,
        "FPN": smp.FPN,
        "DeepLabV3": smp.DeepLabV3,
        "LinkNet": smp.Linknet,
        "UNet++": smp.UnetPlusPlus,
        "DPT": smp.DPT,
        "SegFormer": smp.Segformer,
        "MAnet": smp.MAnet,
        "PAN": smp.PAN,
        "UPerNet": smp.UPerNet,
    }

    model_class = model_map.get(model_name)
    if model_class:
        return model_class(**model_params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

def preprocess(image, device, size=(256, 256)):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def postprocess(pred_mask):
    if pred_mask.ndim == 3:
        pred_mask = np.argmax(pred_mask, axis=0)
    return (pred_mask * 255).astype(np.uint8)

def load_model(config, device):
    print("🚀 Loading model...")
    model = get_model(config).to(device)
    model.load_state_dict(torch.load(config['overplay']['model_path'], map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, config, device, save_path="output.png"):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image, device)

    with torch.no_grad():
        output = model(input_tensor)
        if config['model']['classes'] == 1:
            pred = (torch.sigmoid(output) > 0.5).long()
        else:
            pred = torch.argmax(output, dim=1).long()

    pred_np = pred.squeeze().cpu().numpy()
    mask = postprocess(pred_np)

    Image.fromarray(mask).save(save_path)
    print(f"✅ Saved mask to {save_path}")

def predict_folder(model, folder_path, config, device, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(folder_path):
        if fname.lower().endswith((".jpg", ".png", ".jpeg")):
            in_path = os.path.join(folder_path, fname)
            out_path = os.path.join(output_dir, fname)
            predict_image(model, in_path, config, device, save_path=out_path)

def main(config_path="config.yaml", image_path=None, folder_path=None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = load_model(config, device)

    if image_path is not None:
        predict_image(model, image_path, config, device, save_path="overplay.png")
    elif folder_path is not None:
        predict_folder(model, folder_path, config, device, output_dir="overplay")
    else:
        print("❌ Please provide either an image_path or folder_path")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--image_path", type=str, default=None, help="Path to an image")
    parser.add_argument("--folder_path", type=str, default=None, help="Path to a folder of images")
    args = parser.parse_args()
    main(config_path=args.config, image_path=args.image_path, folder_path=args.folder_path)

    # python overplay.py --image_path test.jpg
    # python overplay.py --folder_path ./test_images