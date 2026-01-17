# Road Segmentation with PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A comprehensive pipeline for training and evaluating road segmentation models on satellite imagery using PyTorch and Segmentation Models PyTorch.

</div>

---

## 📌 Overview

This project provides a complete pipeline for training and evaluating road segmentation models from satellite imagery. Designed with modularity and easy configuration through YAML files, it supports multiple model architectures and encoders.

**Key Features:**
- 🎯 Support for multiple architectures: UNet, UNet++, DeepLabV3Plus,DeepLabV3, LinkNet, DPT, MAnet, PAN, UPerNet, PSPNet, FPN, SegFormer, DPT, EfficientViT-Seg
- 🔧 Flexible configuration via `config.yaml`
- 📊 Track multiple metrics: IoU, F1-score, Accuracy, Dice Loss, Focal Loss
- 💾 Automatic model checkpointing and training results
- 🚀 Automatic GPU/CPU detection
- 📈 Comprehensive visualizations and reports

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Min-1210/road_segmentation.git
cd road_segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Organize your data as follows:

```
Satellite_Datasets/
└── <dataset_name>/
    ├── images/
    │   ├── Train/  # Training images
    │   ├── Val/    # Validation images
    │   └── Test/   # Test images
    └── mask/
        ├── Train/  # Training masks
        ├── Val/    # Validation masks
        └── Test/   # Test masks
```

**Note:** All images must be `.png` files.

### 3. Configure Training

Edit `config.yaml`:

```yaml
data:
  base_dir: "/path/to/Satellite_Datasets"
  dataset_name: "TGRS_Road"  # Your dataset folder name

training:
  batch_size: 16
  num_epochs: 50

model:
  name: "DeepLabV3Plus"  # Model architecture
  encoder_name: "resnet50"  # Backbone encoder
  classes: 2  # Number of classes (2 for binary: road/non-road)

loss:
  name: "CrossEntropyLoss"

optimizer:
  name: "Adam"
  lr: 0.001

scheduler:
  name: "ReduceLROnPlateau"
```

### 4. Train Model

```bash
python train.py
```

Results will be saved to:
- Best model: `model/model_<config_name>.pt`
- Logs: `plot/plot_<config_name>/training.log`
- Metrics: `plot/plot_<config_name>/epoch_results.csv`
- Plots: `plot/plot_<config_name>/training_metrics_summary.png`

### 5. Make Predictions

**Predict a single image:**
```bash
python inference.py \
  --input "/path/to/image.jpg" \
  --weight "model/model_best.pt" \
  --arch "DeepLabV3Plus" \
  --encoder "resnet50" \
  --classes 2
```

**Predict on a folder:**
```bash
python inference.py \
  --input "/path/to/image_folder/" \
  --weight "model/model_best.pt" \
  --arch "DeepLabV3Plus" \
  --encoder "resnet50" \
  --output "predictions"
```

---

## 📁 Project Structure

```
road_segmentation/
├── config.yaml              # Main configuration file
├── dataset.py               # Dataset & DataLoader
├── train.py                 # Training script
├── inference.py             # Inference script
├── test.py                  # Evaluation script
├── utils.py                 # Utility functions
├── plot.py                  # Plotting functions
├── requirements.txt         # Dependencies
├── Satellite_Datasets/      # Data directory
├── model/                   # Trained models
└── plot/                    # Results and plots
```
---

## ⚙️ Advanced Configuration

### Supported Model Architectures

In `config.yaml`, you can choose:

```yaml
model:
  name: "UNet"  # UNet, UNet++, DeepLabV3Plus, FPN, SegFormer, DPT, EfficientViT-Seg
```

### Supported Encoders

```yaml
model:
  encoder_name: "resnet50"
  # Options: resnet18, resnet50, resnet101, efficientnet-b1, 
  # mobileone_s0, vgg11, densenet121, etc.
```

### Loss Functions

```yaml
loss:
  name: "CrossEntropyLoss"
  # Options: CrossEntropyLoss, DiceLoss, JaccardLoss, 
  # FocalLoss, BCEWithLogitsLoss, CombinedLoss
```

### Learning Rate Scheduler

```yaml
scheduler:
  name: "ReduceLROnPlateau"
  params:
    mode: 'min'
    factor: 0.1
    patience: 5
```

---

## 📊 Output Results

After training, the following files are automatically generated:

| File | Description |
|------|-------------|
| `training.log` | Detailed training logs |
| `epoch_results.csv` | Metrics for each epoch |
| `training_metrics_summary.png` | Train/Val metrics plots |
| `confusion_matrix.png` | Confusion matrix visualization |
| `training_times.txt` | Training duration |
| `model_<name>.pt` | Best model (based on Val IoU) |

---

These are my results for the datasets and models: [Result](https://drive.google.com/drive/folders/1Xo9MOrquM-1DjhHSwdEEOqw-q1Iee1i7?usp=sharing)

## 🔧 Evaluate Model

To evaluate your model on the test set:

```bash
python test.py \
  "model/model_best.pt" \
  "Satellite_Datasets/TGRS_Road" \
  --output-dir "test_results/"
```

---

## 💡 Usage Examples

### Example 1: Train with EfficientViT-Seg

```yaml
# config.yaml
model:
  name: "EfficientViT-Seg"
  efficientvit_params:
    model_zoo_name: "efficientvit-seg-l1-ade20k"
    pretrained_seg_weights: "/path/to/weights.pt"
```
When using EfficientViT-Seg, you should add the weights from the [model_zoo_name](https://github.com/mit-han-lab/efficientvit/tree/master/applications/efficientvit_seg) to the assets/efficientvit_seg directory.

```bash
python train.py
```

### Example 2: Train with Multiple Encoders

```bash
python train.py --encoders resnet18 resnet50 mobileone_s0
```

### Example 3: Custom Inference with Different Output

```bash
python inference.py \
  --input "test_images/" \
  --weight "model/best_model.pt" \
  --arch "UNet" \
  --encoder "resnet34" \
  --output "my_predictions/"
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA out of memory** | Reduce `batch_size` in `config.yaml` |
| **Dataset not found** | Check `dataset_name` and `base_dir` in config |
| **Model not saved** | Verify write permissions in `model/` directory |
| **Import errors** | Run `pip install -r requirements.txt` again |
| **Slow training** | Use GPU or reduce `num_epochs` |

---

## 📋 System Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ (CUDA 11.8 recommended for GPU acceleration)
- **RAM**: Minimum 8GB (16GB+ recommended)
- **GPU**: Optional but recommended (10-20x faster)

---

## 📚 References

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientViT](https://github.com/mit-han-lab/efficientvit)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Min-1210** - [GitHub Profile](https://github.com/Min-1210)

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewFeature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/NewFeature`)
5. Open a Pull Request

---

## 📞 Support

If you encounter any issues:
- Check the [Troubleshooting](#-troubleshooting) section
- Open a [GitHub Issue](https://github.com/Min-1210/road_segmentation/issues)
- Review inline code comments

---

**Last Updated**: December 2025
