# road_segmentation
Dự án này sử dụng các mô hình Deep Learning để thực hiện bài toán phân vùng ngữ nghĩa (semantic segmentation), nhằm mục tiêu xác định và khoanh vùng các đối tượng đường đi trong ảnh vệ tinh.
## Tính năng nổi bật ✨
Nhiều kiến trúc mô hình: Dễ dàng chuyển đổi giữa các kiến trúc segmentation phổ biến như Unet, FPN, DeepLabV3, và DeepLabV3+.

Encoder mạnh mẽ: Hỗ trợ các bộ mã hóa (encoder) được tiền huấn luyện trên ImageNet như ResNet50, ResNet101 để tăng hiệu suất.

Đa dạng Hàm Loss: Lựa chọn linh hoạt giữa các hàm loss như DiceLoss, JaccardLoss, BCEWithLogitsLoss, FocalLoss, hoặc một CombinedLoss tùy chỉnh (kết hợp BCE và Dice).

Cấu hình tập trung: Toàn bộ quá trình huấn luyện, từ đường dẫn dữ liệu đến các siêu tham số, đều được quản lý trong một file config.yaml duy nhất.

Đánh giá toàn diện: Tự động tính toán và theo dõi nhiều chỉ số quan trọng như IoU (Jaccard score), F1-score, và pixel accuracy cho cả tập huấn luyện và tập kiểm định.

Trực quan hóa kết quả: Tự động vẽ và lưu lại các biểu đồ chi tiết về quá trình huấn luyện, giúp dễ dàng phân tích và so sánh kết quả.
## Cài đặt 🛠️
Sao chép (Clone) dự án:

Bash

git clone https://github.com/Min-1210/road_segmentation.git

cd your-repository-name

Tạo môi trường ảo (Khuyến khích):

python -m venv venv

source venv/bin/activate  # Trên Windows: venv\Scripts\activate

Sau đó chạy lệnh:

pip install -r requirements.txt

## Cấu trúc Thư mục Dữ liệu 📁
Bạn cần tạo một thư mục chính cho dữ liệu (ví dụ: Satellite_Datasets), và bên trong đó tổ chức các file như sau:

road-segmentation/
├── dataset/                    # Thư mục chứa toàn bộ dữ liệu
│   ├── Train/
│   └── Validation/
│
├── plots/                      # Thư mục lưu biểu đồ và lịch sử huấn luyện
│   ├── training_metrics_summary.png
│   ├── train_losses.npy
│   └── ...
│
├── weights/                    # Thư mục lưu các trọng số đã huấn luyện
│   └── model_refactored.pt
│
├── .gitignore                  # Khai báo các file/thư mục mà Git sẽ bỏ qua
├── config.yaml                 # File cấu hình chính (tham số, đường dẫn, mô hình)
├── dataset.py                  # Lớp Dataset và các hàm tải dữ liệu
├── plot.py                     # Hàm vẽ và lưu biểu đồ kết quả
├── README.md                   # File giới thiệu dự án (hướng dẫn, mô tả)
├── requirements.txt            # Danh sách các thư viện Python cần thiết
├── train.py                    # Script chính để bắt đầu huấn luyện mô hình
└── utils.py                    # Các hàm tiện ích (lấy model, loss, optimizer...)
### Giải thích:
images/: Thư mục chứa tất cả các ảnh gốc.

Train/: Chứa ảnh dùng để huấn luyện.

Val/: Chứa ảnh dùng để kiểm định (validation).

mask/: Thư mục chứa tất cả các ảnh mặt nạ (ground truth).

Train/: Chứa mặt nạ tương ứng với ảnh huấn luyện.

Val/: Chứa mặt nạ tương ứng với ảnh kiểm định.
