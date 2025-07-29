import torch
import random
import numpy as np
import segmentation_models_pytorch as smp


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

    if name == "CombinedLoss":
        return CombinedLoss(**params)
    elif name == "DiceLoss":
        return smp.losses.DiceLoss(mode="binary", **params)
    elif name == "JaccardLoss":
        return smp.losses.JaccardLoss(mode="binary", **params)
    elif name == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif name == "FocalLoss":
        return smp.losses.FocalLoss(mode="binary", **params)
    elif name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(**params)
    else:
        raise ValueError(f"Hàm loss '{name}' không được hỗ trợ.")


def get_model(config):
    model_name = config['model']['name']
    model_params = {k: v for k, v in config['model'].items() if k != 'name'}

    if model_params.get('activation') == 'null':
        model_params['activation'] = None

    if model_name == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(**model_params)
    elif model_name == "Unet":
        return smp.Unet(**model_params)
    elif model_name == "FPN":
        return smp.FPN(**model_params)
    elif model_name == "DeepLabV3":
        return smp.DeepLabV3(**model_params)
    elif model_name == "LinkNet":
        return smp.Linknet(**model_params)
    else:
        raise ValueError(f"Mô hình '{model_name}' không được hỗ trợ.")


def get_optimizer(model, config):
    """Factory để lấy optimizer dựa trên cấu hình."""
    name = config['optimizer']['name']
    lr = config['optimizer']['lr']

    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Optimizer '{name}' không được hỗ trợ.")


def get_scheduler(optimizer, config):
    """Factory để lấy scheduler dựa trên cấu hình."""
    name = config['scheduler']['name']
    params = config['scheduler'].get('params', {})

    if name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        raise ValueError(f"Scheduler '{name}' không được hỗ trợ.")


def pixel_accuracy(outputs, masks):
    """Tính toán độ chính xác pixel, tự động hỗ trợ binary và multiclass."""
    if outputs.shape[1] == 1:
        outputs = torch.sigmoid(outputs)
        preds = (outputs > 0.5)
    else:
        preds = torch.argmax(outputs, dim=1)

    correct = (preds == masks.squeeze(1).long()).float()
    acc = correct.sum() / correct.numel()
    return acc.item()
