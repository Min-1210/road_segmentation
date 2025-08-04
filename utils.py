import torch
import random
import numpy as np
import segmentation_models_pytorch as smp
import logging
import os


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
    }

    model_class = model_map.get(model_name)
    if model_class:
        return model_class(**model_params)
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


def pixel_accuracy(outputs, masks):
    if outputs.shape[1] == 1:  # Binary case
        preds = (torch.sigmoid(outputs) > 0.5)
    else:
        preds = torch.argmax(outputs, dim=1)

    correct = (preds == masks.squeeze(1).long()).float()
    acc = correct.sum() / correct.numel()
    return acc.item()


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
