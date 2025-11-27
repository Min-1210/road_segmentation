# Road Segmentation with PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A complete end-to-end pipeline for training and evaluating road segmentation models on satellite imagery using PyTorch and Segmentation Models PyTorch (SMP).

[English](#road-segmentation-with-pytorch) • [Vietnamese](#gi%E1%BB%9Bi-thi%E1%BB%87u-d%E1%BB%B1-%C3%A1n)

</div>

---

## 📌 Project Overview

This project is a complete pipeline for training and evaluating road segmentation models on satellite images using PyTorch and Segmentation Models PyTorch (SMP). Designed with high modularity and easy configuration, it supports multiple datasets and model architectures, making it extensible for various segmentation tasks.

**Status**: ✅ Active Development  
**Primary Language**: Python (98.5%)  
**Main Framework**: PyTorch 2.0+

---

## ⚡ Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Min-1210/road_segmentation.git
cd road_segmentation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your data in this structure:
```
Satellite_Datasets/
└── DeepGlobal/
    ├── images/
    │   ├── Train/  ├── image1.png
    │   ├── Val/    └── ...
    │   └── Test/
    └── mask/
        ├── Train/  ├── image1.png
        ├── Val/    └── ...
        └── Test/
```

### 3. Train Model

```bash
python train.py  # Uses config.yaml
```

### 4. Make Predictions

```bash
python overplay.py --model_path model/best.pt --image_path image.jpg
```

---

## ✨ Key Features

- **🔧 Flexible Configuration**: All parameters managed in `config.yaml` - no code changes needed
- **🏗️ Multiple Architectures**: UNet, UNet++, DeepLabV3+, FPN, SegFormer, DPT, EfficientViT...
- **⚙️ Multiple Encoders**: ResNet, EfficientNet, MobileOne, VGG, DenseNet, and more
- **📊 Multi-Metric Tracking**: IoU, F1-score, Accuracy, Dice Loss, Focal Loss
- **🤖 Automated Testing**: Batch experiments, hyperparameter tuning, result visualization
- **📈 Detailed Output**: Logs, metrics CSV, confusion matrix, training graphs
- **💾 Smart Checkpointing**: Auto-saves best model based on validation IoU

---

## 📋 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ (with CUDA 11.8 recommended for GPU)
- **RAM**: 8GB minimum (16GB+ recommended)
- **GPU**: Optional but highly recommended (10-20x faster training)

---

## 🗂️ Project Structure

```
road_segmentation/
├── continuous/                 # Experimental/Legacy versions
├── efficientvit.version/        # Lightweight EfficientViT models
├── smp.version/                 # Segmentation Models PyTorch versions
├── Satellite_Datasets/          # Input data directory
│   └── DeepGlobal/
│       ├── images/
│       │   ├── Train/
│       │   ├── Val/
│       │   └── Test/
│       └── mask/
│           ├── Train/
│           ├── Val/
│           └── Test/
├── model/                       # Trained models (.pt files)
├── plot/                        # Results (graphs, logs, confusion matrix)
├── config.yaml                  # Main configuration file
├── dataset.py                   # Dataset & DataLoader classes
├── train.py                     # Single training script
├── train_continuous.py          # Training for batch experiments
├── test.py                      # Evaluation on test set
├── overplay.py                  # Prediction on new images
├── run_parameters.py            # Batch testing multiple encoders
├── plot.py                      # Plotting utilities
├── utils.py                     # Helper functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Usage Guide

### 1️⃣ Configure Training

Edit `config.yaml`:

```yaml
data:
  dataset_name: DeepGlobal      # Your dataset folder name
  batch_size: 16

model:
  name: UNet                    # Architecture
  encoder_name: resnet50        # Backbone encoder

training:
  num_epochs: 50
  learning_rate: 0.001

loss: CrossEntropyLoss
optimizer: Adam
scheduler: CosineAnnealingLR
```

### 2️⃣ Train Single Model

```bash
python train.py
```

**Output locations**:
- Model: `model/model_<config_name>.pt`
- Logs: `plot/plot_<config_name>/training.log`
- Metrics: `plot/plot_<config_name>/epoch_results.csv`
- Graphs: `plot/plot_<config_name>/training_metrics_summary.png`

### 3️⃣ Evaluate on Test Set

```bash
python test.py "model/model_DeepGlobal_UNet++_resnet50.pt" \
               "Satellite_Datasets/DeepGlobal" \
               --output-dir "test_results/"
```

### 4️⃣ Make Predictions

**Single image**:
```bash
python overplay.py --model_path model/best.pt --image_path image.jpg
```

**Entire folder**:
```bash
python overplay.py --model_path model/best.pt --folder_path ./images/
```

### 5️⃣ Batch Testing (Multiple Models)

Edit `run_parameters.py`:

```python
model_name_to_test = "UNet"
encoders_to_test = [
    "resnet50",
    "resnet18",
    "efficientnet-b1",
    "mobileone_s0",
    # Add more encoders...
]
```

Run:
```bash
python run_parameters.py
```

---

## 📊 Output Results

After training, these files are automatically generated:

| File | Description |
|------|-------------|
| `training.log` | Complete training logs |
| `epoch_results.csv` | Per-epoch metrics table |
| `training_metrics_summary.png` | Train/Val metrics graphs |
| `confusion_matrix.png` | Best epoch confusion matrix |
| `training_times.txt` | Training time report |
| `model_<name>.pt` | Best model (based on Val IoU) |

---

## 📝 Project Versions

### 📦 `smp.version/`
- **Purpose**: Segmentation Models PyTorch library
- **Best for**: Rapid prototyping, multiple architectures
- **Supported**: UNet, DeepLabV3+, FPN, SegFormer, DPT

### 🚀 `efficientvit.version/`
- **Purpose**: Optimized EfficientViT architecture
- **Best for**: Lightweight models, edge deployment
- **Advantage**: Fast inference, low memory usage

### 📝 `continuous/`
- **Purpose**: Experimental/legacy implementations
- **Status**: Not actively maintained

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `batch_size` in `config.yaml` |
| **Dataset not found** | Check folder name matches `dataset_name` in config |
| **Model not saving** | Verify write permissions in `model/` directory |
| **Import errors** | Run `pip install -r requirements.txt` again |
| **Slow training** | Use GPU or reduce `num_epochs` |
| **Poor predictions** | Check input/output channels in config match your data |

---

## 📚 References

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientViT](# Road Segmentation with PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

A complete end-to-end pipeline for training and evaluating road segmentation models on satellite imagery using PyTorch and Segmentation Models PyTorch (SMP).

[English](#road-segmentation-with-pytorch) • [Vietnamese](#gi%E1%BB%9Bi-thi%E1%BB%87u-d%E1%BB%B1-%C3%A1n)

</div>

---

## 📌 Project Overview

This project is a complete pipeline for training and evaluating road segmentation models on satellite images using PyTorch and Segmentation Models PyTorch (SMP). Designed with high modularity and easy configuration, it supports multiple datasets and model architectures, making it extensible for various segmentation tasks.

**Status**: ✅ Active Development  
**Primary Language**: Python (98.5%)  
**Main Framework**: PyTorch 2.0+

---

## ⚡ Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Min-1210/road_segmentation.git
cd road_segmentation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your data in this structure:
```
Satellite_Datasets/
└── DeepGlobal/
    ├── images/
    │   ├── Train/  ├── image1.png
    │   ├── Val/    └── ...
    │   └── Test/
    └── mask/
        ├── Train/  ├── image1.png
        ├── Val/    └── ...
        └── Test/
```

### 3. Train Model

```bash
python train.py  # Uses config.yaml
```

### 4. Make Predictions

```bash
python overplay.py --model_path model/best.pt --image_path image.jpg
```

---

## ✨ Key Features

- **🔧 Flexible Configuration**: All parameters managed in `config.yaml` - no code changes needed
- **🏗️ Multiple Architectures**: UNet, UNet++, DeepLabV3+, FPN, SegFormer, DPT, EfficientViT...
- **⚙️ Multiple Encoders**: ResNet, EfficientNet, MobileOne, VGG, DenseNet, and more
- **📊 Multi-Metric Tracking**: IoU, F1-score, Accuracy, Dice Loss, Focal Loss
- **🤖 Automated Testing**: Batch experiments, hyperparameter tuning, result visualization
- **📈 Detailed Output**: Logs, metrics CSV, confusion matrix, training graphs
- **💾 Smart Checkpointing**: Auto-saves best model based on validation IoU

---

## 📋 Requirements

- **Python**: 3.8 or higher
- **PyTorch**: 2.0+ (with CUDA 11.8 recommended for GPU)
- **RAM**: 8GB minimum (16GB+ recommended)
- **GPU**: Optional but highly recommended (10-20x faster training)

---

## 🗂️ Project Structure

```
road_segmentation/
├── continuous/                 # Experimental/Legacy versions
├── efficientvit.version/        # Lightweight EfficientViT models
├── smp.version/                 # Segmentation Models PyTorch versions
├── Satellite_Datasets/          # Input data directory
│   └── DeepGlobal/
│       ├── images/
│       │   ├── Train/
│       │   ├── Val/
│       │   └── Test/
│       └── mask/
│           ├── Train/
│           ├── Val/
│           └── Test/
├── model/                       # Trained models (.pt files)
├── plot/                        # Results (graphs, logs, confusion matrix)
├── config.yaml                  # Main configuration file
├── dataset.py                   # Dataset & DataLoader classes
├── train.py                     # Single training script
├── train_continuous.py          # Training for batch experiments
├── test.py                      # Evaluation on test set
├── overplay.py                  # Prediction on new images
├── run_parameters.py            # Batch testing multiple encoders
├── plot.py                      # Plotting utilities
├── utils.py                     # Helper functions
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🚀 Usage Guide

### 1️⃣ Configure Training

Edit `config.yaml`:

```yaml
data:
  dataset_name: DeepGlobal      # Your dataset folder name
  batch_size: 16

model:
  name: UNet                    # Architecture
  encoder_name: resnet50        # Backbone encoder

training:
  num_epochs: 50
  learning_rate: 0.001

loss: CrossEntropyLoss
optimizer: Adam
scheduler: CosineAnnealingLR
```

### 2️⃣ Train Single Model

```bash
python train.py
```

**Output locations**:
- Model: `model/model_<config_name>.pt`
- Logs: `plot/plot_<config_name>/training.log`
- Metrics: `plot/plot_<config_name>/epoch_results.csv`
- Graphs: `plot/plot_<config_name>/training_metrics_summary.png`

### 3️⃣ Evaluate on Test Set

```bash
python test.py "model/model_DeepGlobal_UNet++_resnet50.pt" \
               "Satellite_Datasets/DeepGlobal" \
               --output-dir "test_results/"
```

### 4️⃣ Make Predictions

**Single image**:
```bash
python overplay.py --model_path model/best.pt --image_path image.jpg
```

**Entire folder**:
```bash
python overplay.py --model_path model/best.pt --folder_path ./images/
```

### 5️⃣ Batch Testing (Multiple Models)

Edit `run_parameters.py`:

```python
model_name_to_test = "UNet"
encoders_to_test = [
    "resnet50",
    "resnet18",
    "efficientnet-b1",
    "mobileone_s0",
    # Add more encoders...
]
```

Run:
```bash
python run_parameters.py
```

---

## 📊 Output Results

After training, these files are automatically generated:

| File | Description |
|------|-------------|
| `training.log` | Complete training logs |
| `epoch_results.csv` | Per-epoch metrics table |
| `training_metrics_summary.png` | Train/Val metrics graphs |
| `confusion_matrix.png` | Best epoch confusion matrix |
| `training_times.txt` | Training time report |
| `model_<name>.pt` | Best model (based on Val IoU) |

---

## 📝 Project Versions

### 📦 `smp.version/`
- **Purpose**: Segmentation Models PyTorch library
- **Best for**: Rapid prototyping, multiple architectures
- **Supported**: UNet, DeepLabV3+, FPN, SegFormer, DPT

### 🚀 `efficientvit.version/`
- **Purpose**: Optimized EfficientViT architecture
- **Best for**: Lightweight models, edge deployment
- **Advantage**: Fast inference, low memory usage

### 📝 `continuous/`
- **Purpose**: Experimental/legacy implementations
- **Status**: Not actively maintained

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| **CUDA out of memory** | Reduce `batch_size` in `config.yaml` |
| **Dataset not found** | Check folder name matches `dataset_name` in config |
| **Model not saving** | Verify write permissions in `model/` directory |
| **Import errors** | Run `pip install -r requirements.txt` again |
| **Slow training** | Use GPU or reduce `num_epochs` |
| **Poor predictions** | Check input/output channels in config match your data |

---

## 📚 References

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientViT](https://github.com/microsoft/Cream/tree/main/EfficientViT)
- [Image Segmentation Metrics](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Min-1210** - [GitHub Profile](https://github.com/Min-1210)

---

## 💝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

If you encounter any issues:

- Check [Troubleshooting](#-troubleshooting) section
- Open a [GitHub Issue](https://github.com/Min-1210/road_segmentation/issues)
- Review code comments and docstrings

---

## 🎯 Roadmap

- [ ] Support for additional datasets (AerialImageDataset, Inria Aerial)
- [ ] Real-time inference API
- [ ] Web demo application
- [ ] Distributed training support
- [ ] Model optimization (quantization, pruning)
- [ ] Export to ONNX and TensorFlow

---

**Last Updated**: November 27, 2025)
- [Image Segmentation Metrics](https://www.jeremyjordan.me/evaluating-image-segmentation-models/)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Min-1210** - [GitHub Profile](https://github.com/Min-1210)

---

## 💝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📞 Support

If you encounter any issues:

- Check [Troubleshooting](#-troubleshooting) section
- Open a [GitHub Issue](https://github.com/Min-1210/road_segmentation/issues)
- Review code comments and docstrings

---

## 🎯 Roadmap

- [ ] Support for additional datasets (AerialImageDataset, Inria Aerial)
- [ ] Real-time inference API
- [ ] Web demo application
- [ ] Distributed training support
- [ ] Model optimization (quantization, pruning)
- [ ] Export to ONNX and TensorFlow

---

**Last Updated**: November 27, 2025
