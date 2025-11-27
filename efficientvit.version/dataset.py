import glob
import os
from typing import Optional, Sequence, List

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import PILToTensor
from PIL import Image
import torch


class CustomDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[str],
        mask_paths: Sequence[str],
        image_transform: Optional[transforms.Compose] = None,
        mask_transform: Optional[transforms.Compose] = None,
        config: Optional[dict] = None,
    ):
        if config is None:
            raise ValueError("Config must be provided to CustomDataset")

        if len(image_paths) != len(mask_paths):
            raise ValueError(
                f"Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) must match."
            )

        self.image_paths: List[str] = list(image_paths)
        self.mask_paths: List[str] = list(mask_paths)
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.num_classes = int(config["model"]["classes"])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Grayscale

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)
        else:
            mask = PILToTensor()(mask)

        if self.num_classes == 1:
            mask = (mask > 127).float()
        else:
            mask = (mask > 127).long() 
        mask = mask.squeeze(0)

        return image, mask


def get_dataloaders(config: dict):
    base_dir = config["data"]["base_dir"]
    dataset_name = config["data"]["dataset_name"]
    dataset_root = os.path.join(base_dir, dataset_name)

    train_image_paths = sorted(
        glob.glob(os.path.join(dataset_root, "images", "Train", "*.png"))
    )
    train_mask_paths = sorted(
        glob.glob(os.path.join(dataset_root, "mask", "Train", "*.png"))
    )
    val_image_paths = sorted(
        glob.glob(os.path.join(dataset_root, "images", "Val", "*.png"))
    )
    val_mask_paths = sorted(
        glob.glob(os.path.join(dataset_root, "mask", "Val", "*.png"))
    )

    if len(train_image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in train directory. "
            f"Path checked: {os.path.join(dataset_root, 'images', 'Train')}"
        )
    if len(val_image_paths) == 0:
        raise FileNotFoundError(
            f"No images found in val directory. "
            f"Path checked: {os.path.join(dataset_root, 'images', 'Val')}"
        )

    assert len(train_image_paths) == len(train_mask_paths), (
    f"Number of train images ({len(train_image_paths)}) and masks ({len(train_mask_paths)}) must match."
)
    assert len(val_image_paths) == len(val_mask_paths), (
    f"Number of val images ({len(val_image_paths)}) and masks ({len(val_mask_paths)}) must match."
)

    print(
        f"🔍 Found {len(train_image_paths)} train images and {len(val_image_paths)} val images "
        f"in the '{dataset_name}' dataset."
    )

    train_image_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_image_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    mask_transform = PILToTensor()

    train_dataset = CustomDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        image_transform=train_image_transform,
        mask_transform=mask_transform,
        config=config,
    )

    val_dataset = CustomDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        image_transform=val_image_transform,
        mask_transform=mask_transform,
        config=config,
    )

    batch_size = config["training"]["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    return train_loader, val_loader
