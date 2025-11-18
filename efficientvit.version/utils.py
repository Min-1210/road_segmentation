import torch
import random
import numpy as np
import segmentation_models_pytorch as smp
import logging
import os
import torch.nn.functional as F
from efficientvit.seg_model_zoo import create_efficientvit_seg_model


def set_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CombinedLoss(torch.nn.Module):

    def __init__(self, weight_bce=0.5, weight_dice=0.5, smooth=1.0):
        super(CombinedLoss, self).__init__()
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode="binary", smooth=smooth)
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, outputs, masks):
        bce = self.bce_loss(outputs, masks)
        dice = self.dice_loss(outputs, masks)
        return self.weight_bce * bce + self.weight_dice * dice

def get_loss_function(config):
    name = config['loss']['name']
    params = config['loss'].get('params', {})

    num_classes = config['model']['classes']
    loss_mode = "binary" if num_classes == 1 else "multiclass"

    if name == "CombinedLoss":
        return CombinedLoss(**params)
    elif name == "DiceLoss":
        return smp.losses.DiceLoss(mode=loss_mode, **params)
    elif name == "JaccardLoss":
        return smp.losses.JaccardLoss(mode=loss_mode, **params)
    elif name == "BCEWithLogitsLoss":
        if num_classes > 1:
            raise ValueError("BCEWithLogitsLoss can only be used with classes=1.")
        return torch.nn.BCEWithLogitsLoss()
    elif name == "FocalLoss":
        return smp.losses.FocalLoss(mode=loss_mode, **params)
    elif name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(**params)
    else:
        raise ValueError(f"Loss function '{name}' is not supported.")


def get_model(config):

    model_name = config['model']['name']

    if model_name == "EfficientViT-Seg":
        params = config['model'].get('efficientvit_params', {})
        num_classes = config['model']['classes']

        model = create_efficientvit_seg_model(
            name=params.get("model_zoo_name"),
            pretrained=False,
            num_classes=num_classes
        )
        
        seg_weights_path = params.get("pretrained_seg_weights")
        if seg_weights_path:
            print(f"Loading pretrained segmentation weights from: {seg_weights_path}")
            try:
                checkpoint = torch.load(seg_weights_path, map_location="cpu")
                if 'state_dict' in checkpoint:
                    full_state_dict = checkpoint['state_dict']
                    print("INFO: Loaded 'state_dict' from within a checkpoint object.")
                else:
                    full_state_dict = checkpoint
                    print("INFO: Loaded a raw state_dict file.")
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                skipped_layers = []
                for k, v in full_state_dict.items():
                    if k in model_state_dict:
                        if model_state_dict[k].shape == v.shape:
                            filtered_state_dict[k] = v
                        else:
                            skipped_layers.append(k)
                    else:
                        skipped_layers.append(k)
                model.load_state_dict(filtered_state_dict, strict=False)
                print("✅ Successfully loaded weights (strict=False).")
                if skipped_layers:
                    print(f"⚠️ Skipped layers (due to mismatch): {skipped_layers}")
            except Exception as e:
                print(f"❌ ERROR loading pretrained weights: {e}")
                print("Continuing with randomly initialized model.")
        return model

    smp_params = {
        'encoder_name': config['model'].get('encoder_name', 'resnet34'),
        'encoder_weights': config['model'].get('encoder_weights', 'imagenet'),
        'in_channels': config['model'].get('in_channels', 3),
        'classes': config['model'].get('classes', 1),
        'activation': config['model'].get('activation', None),
    }
    if smp_params.get('activation') == 'null':
        smp_params['activation'] = None

    model_map = {
        "DeepLabV3Plus": smp.DeepLabV3Plus, "Unet": smp.Unet, "FPN": smp.FPN,
        "DeepLabV3": smp.DeepLabV3, "LinkNet": smp.Linknet, "UNet++": smp.UnetPlusPlus,
        "DPT": smp.DPT, "SegFormer": smp.Segformer, "MAnet": smp.MAnet,
        "PAN": smp.PAN, "UPerNet": smp.UPerNet, "PSPNet": smp.PSPNet,
    }

    model_class = model_map.get(model_name)
    if model_class:
        return model_class(**smp_params)
    else:
        raise ValueError(f"Model '{model_name}' is not supported.")

def get_optimizer(model, config):
    name = config['optimizer']['name']
    lr = config['optimizer']['lr']

    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer '{name}' is not supported.")


def get_scheduler(optimizer, config):
    name = config['scheduler']['name']
    params = config['scheduler'].get('params', {})

    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        raise ValueError(f"Scheduler '{name}' is not supported.")


def pixel_accuracy(outputs, masks, threshold: float = 0.5):
    if outputs.dim() != 4:
        raise ValueError(f"outputs must be 4D (B,C,H,W), got {outputs.shape}")

    if masks.dim() == 4:
        if masks.size(1) == 1:
            masks = masks[:, 0, ...]
        else:
            masks = masks.argmax(dim=1)
    elif masks.dim() == 3:
        pass
    else:
        raise ValueError(f"masks must be 3D or 4D, got {masks.shape}")

    if outputs.shape[-2:] != masks.shape[-2:]:
        outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)

    if outputs.size(1) == 1:
        logits = outputs[:, 0, ...]
        preds = (torch.sigmoid(logits) > threshold).long()
    else:
        preds = torch.argmax(outputs, dim=1).long()

    masks = masks.long()
    correct = (preds == masks).float()
    return (correct.sum() / correct.numel()).item()

def setup_logging(config):
    log_dir = config['training']['plot_dir']
    log_file = os.path.join(log_dir, 'training.log')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
