import glob
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_transform=None, mask_transform=None, config=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        if config is None:
            raise ValueError("Config must be provided to CustomDataset")
        self.num_classes = config['model']['classes']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L") # Grayscale

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.num_classes == 1:
            mask = (mask > 0).float() # Binary mask [0, 1]
        else:
            mask = (mask > 0).long().squeeze(0) # Multiclass mask [0, 1, 2...]

        return image, mask


def get_dataloaders(config):
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.ToTensor()

    base_dir = config['data']['base_dir']
    dataset_name = config['data']['dataset_name']
    dataset_root = os.path.join(base_dir, dataset_name)

    train_image_paths = sorted(glob.glob(os.path.join(dataset_root, "images/Train/*.png")))
    train_mask_paths = sorted(glob.glob(os.path.join(dataset_root, "mask/Train/*.png")))
    val_image_paths = sorted(glob.glob(os.path.join(dataset_root, "images/Val/*.png")))
    val_mask_paths = sorted(glob.glob(os.path.join(dataset_root, "mask/Val/*.png")))

    assert len(train_image_paths) > 0, f"No images found in train directory! Path checked: {os.path.join(dataset_root, 'images/Train/*.png')}"
    assert len(train_image_paths) == len(train_mask_paths), "Number of train images and masks do not match!"
    assert len(val_image_paths) > 0, f"No images found in val directory! Path checked: {os.path.join(dataset_root, 'images/Val/*.png')}"
    assert len(val_image_paths) == len(val_mask_paths), "Number of val images and masks do not match!"

    print(f"🔍 Found {len(train_image_paths)} images in the train set and {len(val_image_paths)} images in the val set from the '{dataset_name}' dataset.")

    train_dataset = CustomDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
        config=config
    )
    val_dataset = CustomDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
        config=config
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False, num_workers=0, drop_last=False
    )

    return train_loader, val_loader
