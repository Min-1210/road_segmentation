# Phân Đoạn Đường Bộ với PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Pipeline hoàn chỉnh để huấn luyện và đánh giá các mô hình phân đoạn đường bộ trên ảnh vệ tinh sử dụng PyTorch và Segmentation Models PyTorch.

</div>

---

## 📌 Tổng Quan

Dự án này cung cấp một pipeline đầy đủ để huấn luyện và đánh giá các mô hình phân đoạn đường bộ (road segmentation) từ ảnh vệ tinh. Thiết kế modular và dễ dàng cấu hình thông qua file YAML, hỗ trợ nhiều kiến trúc mô hình và encoder khác nhau.

**Đặc điểm chính:**
- 🎯 Hỗ trợ nhiều kiến trúc: UNet, UNet++, DeepLabV3+, FPN, SegFormer, DPT, EfficientViT-Seg
- 🔧 Cấu hình linh hoạt qua file `config.yaml`
- 📊 Theo dõi nhiều chỉ số: IoU, F1-score, Accuracy, Dice Loss, Focal Loss
- 💾 Tự động lưu mô hình tốt nhất và kết quả huấn luyện
- 🚀 Hỗ trợ GPU/CPU tự động phát hiện
- 📈 Visualizations và báo cáo chi tiết

---

## 🚀 Bắt Đầu Nhanh

### 1. Cài Đặt

```bash
# Clone repository
git clone https://github.com/Min-1210/road_segmentation.git
cd road_segmentation

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/macOS
# hoặc
venv\\Scripts\\activate  # Windows

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Chuẩn Bị Dữ Liệu

Tổ chức dữ liệu theo cấu trúc sau:

```
Satellite_Datasets/
└── <tên_dataset>/
    ├── images/
    │   ├── Train/  # Ảnh huấn luyện
    │   ├── Val/    # Ảnh validation
    │   └── Test/   # Ảnh test
    └── mask/
        ├── Train/  # Mask huấn luyện
        ├── Val/    # Mask validation
        └── Test/   # Mask test
```

**Lưu ý:** Tất cả ảnh phải là file `.png`

### 3. Cấu Hình Huấn Luyện

Chỉnh sửa file `config.yaml`:

```yaml
data:
  base_dir: "/đường/dẫn/đến/Satellite_Datasets"
  dataset_name: "TGRS_Road"  # Tên thư mục dataset của bạn

training:
  batch_size: 16
  num_epochs: 50

model:
  name: "DeepLabV3Plus"  # Kiến trúc model
  encoder_name: "resnet50"  # Backbone encoder
  classes: 2  # Số lớp (2 cho binary: đường/không phải đường)

loss:
  name: "CrossEntropyLoss"

optimizer:
  name: "Adam"
  lr: 0.001

scheduler:
  name: "ReduceLROnPlateau"
```

### 4. Huấn Luyện

```bash
python train.py
```

Kết quả sẽ được lưu tại:
- Model tốt nhất: `model/model_<config_name>.pt`
- Logs: `plot/plot_<config_name>/training.log`
- Metrics: `plot/plot_<config_name>/epoch_results.csv`
- Biểu đồ: `plot/plot_<config_name>/training_metrics_summary.png`

### 5. Dự Đoán

**Dự đoán một ảnh:**
```bash
python inference.py \
  --input "/đường/dẫn/ảnh.jpg" \
  --weight "model/model_best.pt" \
  --arch "DeepLabV3Plus" \
  --encoder "resnet50" \
  --classes 2
```

**Dự đoán cả thư mục:**
```bash
python inference.py \
  --input "/đường/dẫn/thư_mục_ảnh/" \
  --weight "model/model_best.pt" \
  --arch "DeepLabV3Plus" \
  --encoder "resnet50" \
  --output "predictions"
```

---

## 📁 Cấu Trúc Dự Án

```
road_segmentation/
├── config.yaml              # File cấu hình chính
├── dataset.py               # Dataset & DataLoader
├── train.py                 # Script huấn luyện
├── inference.py             # Script dự đoán
├── test.py                  # Script đánh giá
├── utils.py                 # Các hàm tiện ích
├── plot.py                  # Vẽ biểu đồ
├── requirements.txt         # Thư viện cần thiết
├── Satellite_Datasets/      # Thư mục chứa dữ liệu
├── model/                   # Mô hình đã huấn luyện
└── plot/                    # Kết quả và biểu đồ
```

---

## ⚙️ Cấu Hình Chi Tiết

### Các Kiến Trúc Model Được Hỗ Trợ

Trong `config.yaml`, bạn có thể chọn:

```yaml
model:
  name: "UNet"  # UNet, UNet++, DeepLabV3Plus, FPN, SegFormer, DPT, EfficientViT-Seg
```

### Các Encoder Được Hỗ Trợ

```yaml
model:
  encoder_name: "resnet50"
  # Lựa chọn: resnet18, resnet50, resnet101, efficientnet-b1, 
  # mobileone_s0, vgg11, densenet121, v.v.
```

### Các Loss Function

```yaml
loss:
  name: "CrossEntropyLoss"
  # Lựa chọn: CrossEntropyLoss, DiceLoss, JaccardLoss, 
  # FocalLoss, BCEWithLogitsLoss, CombinedLoss
```

### Scheduler

```yaml
scheduler:
  name: "ReduceLROnPlateau"
  params:
    mode: 'min'
    factor: 0.1
    patience: 5
```

---

## 📊 Kết Quả Đầu Ra

Sau khi huấn luyện, các file sau sẽ được tạo tự động:

| File | Mô tả |
|------|-------|
| `training.log` | Log chi tiết quá trình huấn luyện |
| `epoch_results.csv` | Bảng metrics theo từng epoch |
| `training_metrics_summary.png` | Biểu đồ metrics train/val |
| `confusion_matrix.png` | Ma trận nhầm lẫn |
| `training_times.txt` | Thời gian huấn luyện |
| `model_<name>.pt` | Model tốt nhất (dựa trên Val IoU) |

---

## 🔧 Đánh Giá Model

Để đánh giá model trên tập test:

```bash
python test.py \
  "model/model_best.pt" \
  "Satellite_Datasets/TGRS_Road" \
  --output-dir "test_results/"
```

---

## 💡 Ví Dụ Sử Dụng

### Ví dụ 1: Huấn luyện với EfficientViT-Seg

```yaml
# config.yaml
model:
  name: "EfficientViT-Seg"
  efficientvit_params:
    model_zoo_name: "efficientvit-seg-l1-ade20k"
    pretrained_seg_weights: "đường/dẫn/weights.pt"
```

```bash
python train.py
```

### Ví dụ 2: Huấn luyện nhiều encoder

```bash
python train.py --encoders resnet18 resnet50 mobileone_s0
```

### Ví dụ 3: Dự đoán với output tùy chỉnh

```bash
python inference.py \
  --input "test_images/" \
  --weight "model/best_model.pt" \
  --arch "UNet" \
  --encoder "resnet34" \
  --output "my_predictions/"
```

---

## 🛠️ Xử Lý Sự Cố

| Vấn đề | Giải pháp |
|--------|----------|
| **CUDA out of memory** | Giảm `batch_size` trong `config.yaml` |
| **Không tìm thấy dataset** | Kiểm tra `dataset_name` và `base_dir` trong config |
| **Model không lưu** | Kiểm tra quyền ghi trong thư mục `model/` |
| **Import error** | Chạy lại `pip install -r requirements.txt` |
| **Huấn luyện chậm** | Sử dụng GPU hoặc giảm `num_epochs` |

---

## 📋 Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **PyTorch**: 2.0+ (khuyến nghị CUDA 11.8 cho GPU)
- **RAM**: Tối thiểu 8GB (khuyến nghị 16GB+)
- **GPU**: Tùy chọn nhưng khuyến nghị (nhanh hơn 10-20 lần)

---

## 📚 Tài Liệu Tham Khảo

- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [EfficientViT](https://github.com/mit-han-lab/efficientvit)

---

## 📄 Giấy Phép

MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.

---

## 👤 Tác Giả

**Min-1210** - [GitHub Profile](https://github.com/Min-1210)

---

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/TinhNangMoi`)
3. Commit thay đổi (`git commit -m 'Thêm tính năng mới'`)
4. Push lên branch (`git push origin feature/TinhNangMoi`)
5. Mở Pull Request

---

## 📞 Hỗ Trợ

Nếu gặp vấn đề:
- Kiểm tra phần [Xử Lý Sự Cố](#-xử-lý-sự-cố)
- Mở [GitHub Issue](https://github.com/Min-1210/road_segmentation/issues)
- Đọc comments trong code

---

**Cập nhật lần cuối**: Tháng 12, 2025
