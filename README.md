# road_segmentation
Dự án này là một quy trình hoàn chỉnh để huấn luyện và đánh giá các mô hình phân vùng ảnh (Image Segmentation) sử dụng PyTorch và thư viện Segmentation Models PyTorch (SMP). Project được thiết kế để có tính module hóa cao, dễ dàng cấu hình và mở rộng cho các bộ dữ liệu và kiến trúc mô hình khác nhau, đặc biệt là ảnh vệ tinh.

## Tính năng nổi bật ✨
Cấu hình linh hoạt: Toàn bộ tham số được quản lý tập trung trong file config.yaml, giúp việc thử nghiệm trở nên dễ dàng mà không cần sửa code.

Hỗ trợ nhiều kiến trúc: Dễ dàng chuyển đổi giữa các kiến trúc segmentation phổ biến (UNet, UNet++, DeepLabV3+, FPN...) và các encoder khác nhau (ResNet, EfficientNet, MobileOne...) từ thư viện SMP.

Quy trình huấn luyện toàn diện:

Theo dõi đa chỉ số (IoU, F1-score, Accuracy, Dice Loss, Focal Loss).

Lưu lại model có kết quả tốt nhất trên tập validation.

Tự động tạo biểu đồ, ma trận nhầm lẫn (confusion matrix) và file log chi tiết (.csv, .log, .txt).

Tự động hóa thử nghiệm: Script run_parameters.py cho phép tự động huấn luyện và đánh giá hàng loạt mô hình với các encoder khác nhau.

Các kịch bản sử dụng đầy đủ: Cung cấp các script riêng biệt cho Huấn luyện, Đánh giá, và Dự đoán trên ảnh mới.

## Cấu trúc dự án
```bash
.
├── Satellite_Datasets/       # Thư mục chứa các bộ dữ liệu
│   └── DeepGlobal/
│       ├── images/
│       │   ├── Train/
│       │   └── Val/
│       │   └── Test/
│       └── mask/
│           ├── Train/
│           └── Val/
│           └── Test/
├── model/                    # Thư mục chứa các file model đã huấn luyện (.pt)
├── plot/                     # Thư mục chứa kết quả (biểu đồ, log, ma trận nhầm lẫn)
├── config.yaml               # File cấu hình chính của dự án
├── dataset.py                # Định nghĩa lớp Dataset và DataLoader
├── overplay.py               # Script để chạy dự đoán trên ảnh mới
├── plot.py                   # (Hàm vẽ biểu đồ phụ trợ)
├── run_parameters.py         # Script tự động chạy nhiều thử nghiệm
├── test.py                   # Script đánh giá model trên tập test
├── train.py                  # Script huấn luyện chính (chạy đơn lẻ)
├── train_continuous.py       # Script huấn luyện (dùng cho run_parameters.py)
├── utils.py                  # Các hàm tiện ích (lấy model, loss, optimizer...)
└── README.md                 # File hướng dẫn
```
## Sử dụng 🛠️
1. Chuẩn bị môi trường:
   Clone repository này về máy:
```bash
git clone https://github.com/Min-1210/road_segmentation.git
```
```bash
cd road_segmentation
```
    Tạo  
2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```

3. Thay đổi tham số ở file config

4. Chạy file train
```bash
python train.py
```
## Sử dụng Continuous 🛠️
Sử dụng để train một kiến trúc với nhiều backbone không cần thay đổi sau mỗi lần train xong

1. Tải source code về máy tính của bạn
```bash
git clone https://github.com/Min-1210/road_segmentation.git
```

2. Cài đặt thư viện
```bash
pip install -r requirements.txt
```
3. Chuyển train_continuous.py và run_parameters.py từ Continuous ra ngoài không gian làm việc

4. Thay đổi file config.yaml

Có thể thay đổi các thông số trừ model: name và encoder_name còn lại có thể thay đổi được

5. Thay đổi thông số trong file run_parameters.py

```bash
    model_name_to_test = "SegFormer"
    encoders_to_test = [
        "resnet50",  # Thông số cần thay đổi
        "resnet18",
        "vgg11",
        "vgg13",
        "densenet121",
        "densenet169",
        "efficientnet-b1",
        "efficientnet-b2",
        "efficientnet-b3",
        "mobileone_s0",
        "mobileone_s1",
        "mobileone_s2",
        "mobileone_s3",
        "mobileone_s4",
    ]
```
Thay đổi "model_name_to_test" với các kiến trúc như: Unet, FPN, DeepLabV3, DeepLabV3Plus, Unet++, DPT, SegFormer

Thay đổi "encoders_to_test" với các backbone muốn sử dụng

6. Train model
```bash
python run_parameters.py
```


