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

git clone [https://github.com/your-username/your-repository-name.git](https://github.com/Min-1210/road_segmentation.git)

cd your-repository-name

Tạo môi trường ảo (Khuyến khích):

python -m venv venv

source venv/bin/activate  # Trên Windows: venv\Scripts\activate

Sau đó chạy lệnh:

pip install -r requirements.txt

## Cấu trúc Thư mục Dữ liệu 📁
Bạn cần tạo một thư mục chính cho dữ liệu (ví dụ: Satellite_Datasets), và bên trong đó tổ chức các file như sau:

Tên_Dự_Án/
├── Satellite_Datasets/

│   └── TGRS_Road/

│       ├── images/
│       │   ├── Train/

│       │   │   ├── 0001.png

│       │   │   ├── 0002.png

│       │   │   └── ...
│       │   └── Val/

│       │       ├── 0100.png
│       │       ├── 0101.png
│       │       └── ...
│       │
│       └── mask/
│           ├── Train/
│           │   ├── 0001.png  # Tên file phải khớp với file ảnh
│           │   ├── 0002.png
│           │   └── ...
│           └── Val/
│               ├── 0100.png  # Tên file phải khớp với file ảnh
│               ├── 0101.png
│               └── ...
│
├── config.yaml
├── train.py
└── ... (các file .py khác)
### Giải thích:
images/: Thư mục chứa tất cả các ảnh gốc.

Train/: Chứa ảnh dùng để huấn luyện.

Val/: Chứa ảnh dùng để kiểm định (validation).

mask/: Thư mục chứa tất cả các ảnh mặt nạ (ground truth).

Train/: Chứa mặt nạ tương ứng với ảnh huấn luyện.

Val/: Chứa mặt nạ tương ứng với ảnh kiểm định.
