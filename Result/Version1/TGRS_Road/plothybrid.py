import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1 or len(x) < 2:
        return x
    k = min(k, len(x))  # tránh kernel lớn hơn dữ liệu
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="valid")

def safe_load_array(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception as e:
        print(f"⚠️  Không thể load '{path}': {e}")
        return None

def create_comparison_plot(models_to_plot, output_filename,
                           title="Training Performance Comparison of Models TGRS_Road (mobileone_s4)",
                           smooth_window: int | None = None):
    metrics = {
        'Loss Comparison': ('train_loss', 'val_loss'),
        'IoU Score Comparison': ('train_iou_score', 'val_iou_score'),
        'F1-Score Comparison': ('train_f1_score', 'val_f1_score'),
        'Pixel Accuracy Comparison': ('train_accuracy', 'val_accuracy'),
    }

    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # chuẩn bị figure
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    fig.suptitle(title, fontsize=20, y=0.98)

    # dùng color cycle mặc định của matplotlib để không giới hạn số model
    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not prop_cycle:
        prop_cycle = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                      '#bcbd22', '#17becf']
    color_iter = cycle(prop_cycle)
    model_colors = {}

    # Vẽ từng subplot theo metric
    for ax, (title_text, (train_key, val_key)) in zip(axes, metrics.items()):
        ax.set_title(title_text, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        any_plotted = False

        # reset màu cho mỗi metric để model giữ màu nhất quán trong cùng một subplot
        # (tuỳ chọn: muốn mỗi metric đổi màu khác cũng được)
        model_colors.clear()

        for model in models_to_plot:
            name = model['name']
            path = Path(model['path'])
            train_path = path / f"{train_key}.npy"
            val_path = path / f"{val_key}.npy"

            if not train_path.exists() or not val_path.exists():
                print(f"⚠️  Thiếu file .npy cho model '{name}' trong '{path}'. Bỏ qua metric '{title_text}'.")
                continue

            train_arr = safe_load_array(train_path)
            val_arr = safe_load_array(val_path)
            if train_arr is None or val_arr is None:
                continue

            # đồng bộ độ dài nếu lệch
            L = min(len(train_arr), len(val_arr))
            if L == 0:
                print(f"⚠️  Dữ liệu rỗng cho '{name}' - '{title_text}'.")
                continue

            train_arr = train_arr[:L]
            val_arr = val_arr[:L]

            # smoothing (tùy chọn)
            t = np.arange(1, L + 1)
            if smooth_window and smooth_window > 1:
                train_arr_sm = moving_average(train_arr, smooth_window)
                val_arr_sm = moving_average(val_arr, smooth_window)
                # căn lại trục epoch cho chuỗi đã làm mượt
                offset = len(train_arr) - len(train_arr_sm)
                t_sm = t[offset:]
                t_train, y_train = t_sm, train_arr_sm
                t_val, y_val = t_sm, val_arr_sm
            else:
                t_train, y_train = t, train_arr
                t_val, y_val = t, val_arr

            if name not in model_colors:
                model_colors[name] = next(color_iter)
            c = model_colors[name]

            ax.plot(t_train, y_train, linestyle='-',  color=c, label=f"{name} (Train)")
            ax.plot(t_val,   y_val,   linestyle='--', color=c, label=f"{name} (Val)")
            any_plotted = True

        if any_plotted:
            # gọn legend
            ax.legend(ncol=2, fontsize=10)
            ax.grid(True)
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12, alpha=0.7)
            ax.grid(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Lưu PNG và SVG
    png_path = output_path.with_suffix(".png")
    svg_path = output_path.with_suffix(".svg")
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path)  # vector, sắc nét khi chèn báo cáo
    plt.close()

    print(f"✅ Đã lưu biểu đồ:\n - {png_path}\n - {svg_path}")

if __name__ == '__main__':
    models_to_plot = [
        {"name": "DeepLabV3+", "path": r"DeepLabV3+\plot\plot_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s4"},
        {"name": "UNet++",     "path": r"UNet++\plot\plot_TGRS_Road_CrossEntropyLoss_UNet++_mobileone_s4"},
        {"name": "SegFormer",  "path": r"SegFormer\plot\plot_TGRS_Road_CrossEntropyLoss_SegFormer_mobileone_s4"},
        # Có thể thêm nhiều model nữa, không lo hết màu
    ]

    # tên file đầu ra (không cần đuôi – hàm tự lưu .png và .svg)
    output_filename = "model_comparison_summary_mobile_s4"

    # smooth_window: None hoặc 1 để tắt; 5–15 để làm mượt nhẹ
    create_comparison_plot(models_to_plot, output_filename, smooth_window=7)
