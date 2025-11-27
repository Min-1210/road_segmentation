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
## Bắt đầu
1. Chuẩn bị môi trường:
   Clone repository này về máy:
```bash
   git clone https://github.com/Min-1210/road_segmentation.git
```
```bash
   cd road_segmentation
```
   Tạo và kích hoạt môn trường ảo:
```bash
# Dành cho Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Dành cho Windows
python -m venv venv
.\venv\Scripts\activate
```
Môi trường conda:
1.1. Tạo một môi trường mới (ví dụ: segmentation_env) với phiên bản Python phù hợp:
```bash
conda create --name segmentation_env python=3.11 -y
```
1.2. Kích hoạt môi trường vừa tạo
```bash
conda activate segmentation_env
```
   Cài đặt thư viện
```bash
pip install -r requirements.txt
```
2. Chuẩn bị Dataset
```bash
Satellite_Datasets/
└── Tên_Dataset_Của_Bạn/  (ví dụ: DeepGlobal)
    ├── images/
    │   ├── Train/
    │   │   ├── image1.png
    │   │   └── ...
    │   ├── Val/
    │   │   ├── image2.png
    │   │   └── ...
    │   └── Test/ (Tùy chọn, dùng cho test.py)
    │       ├── image3.png
    │       └── ...
    └── mask/
        ├── Train/
        │   ├── image1.png
        │   └── ...
        ├── Val/
        │   ├── image2.png
        │   └── ...
        └── Test/ (Tùy chọn, dùng cho test.py)
            ├── image3.png
            └── ...
```
Lưu ý: Tên của ảnh và mask tương ứng phải giống hệt nhau.

3. Tải checkpoint:

   https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_cityscapes.pt
   
   https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_cityscapes.pt

   https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l1_ade20k.pt

   https://huggingface.co/han-cai/efficientvit-seg/resolve/main/efficientvit_seg_l2_ade20k.pt

Tạo file ```bash efficientvit-seg``` trong ```assets``` rồi thêm các checkpoint vào. 

## Sử dụng ⚙️
Sử dụng để train một kiến trúc với nhiều backbone không cần thay đổi sau mỗi lần train xong
1. Cấu hình thử nghiệm (config.yaml)
Mở file config.yaml và chỉnh sửa các tham số cho phù hợp.

data: Chỉ định tên bộ dữ liệu bạn muốn sử dụng (phải khớp với tên thư mục trong Satellite_Datasets).

model: Chọn name (kiến trúc model) và encoder_name bạn muốn thử nghiệm.

training: Thiết lập batch_size, num_epochs.

loss, optimizer, scheduler: Chọn các hàm và tham số tương ứng.

2. Huấn luyện một mô hình
Để bắt đầu quá trình huấn luyện với cấu hình trong config.yaml, chạy lệnh:
```bash
python train.py
```
Quá trình huấn luyện sẽ bắt đầu. Model tốt nhất (dựa trên Val IoU) và các kết quả sẽ được lưu vào thư mục model/ và plot/ với tên được tạo tự động dựa trên cấu hình.

3. Đánh giá mô hình trên tập Test
Sau khi huấn luyện, bạn có thể đánh giá hiệu năng của model trên tập dữ liệu test.

Ví dụ:
```Bash
python test.py "model/model_DeepGlobal_CrossEntropyLoss_UNet++_mobileone_s4.pt" "Satellite_Datasets/DeepGlobal" --output-dir "test_results/UNet++_s4"
```
4. Dự đoán trên ảnh mới (overplay.py)
Sử dụng script overplay.py để áp dụng model đã huấn luyện lên một ảnh hoặc một thư mục ảnh. Script này được thiết kế để chạy độc lập và tự động nhận diện cấu hình từ tên file model.

Dự đoán một ảnh:
```Bash
python overplay.py --model_path "đường/dẫn/tới/model.pt" --image_path "ảnh/cần/dự/đoán.jpg"
```
Dự đoán cả thư mục:
```Bash
python overplay.py --model_path "đường/dẫn/tới/model.pt" --folder_path "thư/mục/chứa/ảnh"
```
5. Chạy hàng loạt thử nghiệm
Script run_parameters.py là một công cụ mạnh mẽ để tự động huấn luyện và so sánh nhiều encoder cho một kiến trúc model nhất định.

Mở file run_parameters.py.

Chỉnh sửa model_name_to_test và danh sách encoders_to_test.

Chạy script:

```Bash
python run_parameters.py
```
Script sẽ lần lượt chạy qua từng encoder trong danh sách, mỗi lần chạy là một quy trình huấn luyện đầy đủ.

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

#### 📊 Kết quả đầu ra
Sau khi chạy huấn luyện, các file sau sẽ được tự động tạo ra trong thư mục plot/plot_<tên_cấu_hình>:

training.log: Log chi tiết toàn bộ quá trình.

epoch_results.csv: Bảng tổng hợp kết quả của từng epoch.

training_metrics_summary.png: Biểu đồ so sánh các chỉ số train/val.

confusion_matrix.png: Ma trận nhầm lẫn của epoch tốt nhất.

training_times.txt: Báo cáo thời gian huấn luyện.

Model tốt nhất sẽ được lưu tại model/model_<tên_cấu_hình>.pt.

```bash
python demo_efficientvit_seg_model.py \
    --model "efficientvit-seg-l1-ade20k" \
    --weight_path "/home/weed/Pictures/road_segmentation/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l1.pt" \
    --image_path "/home/weed/Pictures/road_segmentation/Satellite_Datasets/TGRS_Road/images/Test/image (65).png" \
    --output_path "/home/weed/Pictures/road_segmentation/prediction.png"
```


