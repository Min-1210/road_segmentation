import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
import re


def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k is None or k <= 1 or len(x) < 2:
        return x
    k = min(k, len(x))
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(x, kernel, mode="valid")


def safe_load_array(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception as e:
        print(f"⚠️  Không thể load '{path}': {e}")
        return None



def create_comparison_plot(models_to_plot, output_filename, title,
                           train_key, val_key, smooth_window=None):
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    fig.suptitle(title, fontsize=20, y=0.95)

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    if not prop_cycle:
        prop_cycle = ['#ff7f0e', '#1f77b4', '#2ca02c', '#d62728',
                      '#9467bd', '#8c564b', '#e377c2', ]
    color_iter = cycle(prop_cycle)
    model_colors = {}

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    any_plotted_on_this_axis = False

    for model in models_to_plot:
        name = model['name']
        path = Path(model['path'])

        if name not in model_colors:
            model_colors[name] = next(color_iter)
        c = model_colors[name]

        if train_key:
            train_path = path / f"{train_key}.npy"
            train_arr = safe_load_array(train_path)

            if train_arr is not None and len(train_arr) > 0:
                L_train = len(train_arr)
                t_train_orig = np.arange(1, L_train + 1)

                if smooth_window and smooth_window > 1:
                    train_arr_sm = moving_average(train_arr, smooth_window)
                    offset = L_train - len(train_arr_sm)
                    t_train, y_train = t_train_orig[offset:], train_arr_sm
                else:
                    t_train, y_train = t_train_orig, train_arr

                ax.plot(t_train, y_train, linestyle='-', color=c, label=f"{name} (Train)")
                any_plotted_on_this_axis = True

        if val_key:
            val_path = path / f"{val_key}.npy"
            val_arr = safe_load_array(val_path)

            if val_arr is not None and len(val_arr) > 0:
                L_val = len(val_arr)
                t_val_orig = np.arange(1, L_val + 1)

                if smooth_window and smooth_window > 1:
                    val_arr_sm = moving_average(val_arr, smooth_window)
                    offset = L_val - len(val_arr_sm)
                    t_val, y_val = t_val_orig[offset:], val_arr_sm
                else:
                    t_val, y_val = t_val_orig, val_arr

                ax.plot(t_val, y_val, linestyle='--', color=c, label=f"{name} (Val)")
                any_plotted_on_this_axis = True

    if any_plotted_on_this_axis:
        # ax.legend(ncol=2, fontsize=10)
        ax.grid(True)
    else:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=12, alpha=0.7)
        ax.grid(False)

    plt.tight_layout(rect=[0, 0.03, 1, 0.90])

    png_path = output_path.with_suffix(".png")
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"✅ Đã lưu biểu đồ:\n - {png_path}")

if __name__ == '__main__':
    models_to_plot = [
        {"name": "DeepLabV3Plus",
         "path": r"DeepLabV3Plus/plot/plot_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1"},
        {"name": "PAN", "path": r"PAN/plot/plot_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s1"},
        {"name": "MAnet", "path": r"MAnet/plot/plot_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s1"},
        {"name": "UPerNet", "path": r"UPerNet/plot/plot_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s1"},
        {"name": "PSPNet", "path": r"PSPNet/plot/plot_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s1"},
    ]

    metrics = {
        'Train Loss Comparison': ('train_loss', None),
        'Validation Loss Comparison': (None, 'val_loss'),
        'Train IoU Score Comparison': ('train_iou_score', None),
        'Validation IoU Score Comparison': (None, 'val_iou_score'),
        'Train Pixel Accuracy Comparison': ('train_accuracy', None),
        'Validation Accuracy Comparison': (None, 'val_accuracy'),
        'Train F1-Score Comparison': ('train_f1_score', None),
        'Validation F1-Score Comparison': (None, 'val_f1_score'),
    }

    base_title_suffix = "of Models \n TGRS_Road (mobileone_s1)"
    base_filename_prefix = "model_comparison_s1"
    smooth_window_setting = None

    metrics_to_run = metrics

    if not metrics_to_run:
        print("❌ Không có metric nào được định nghĩa để vẽ. Hãy bỏ comment trong biến 'metrics'.")
    else:
        print(f"🔍 Tìm thấy {len(metrics_to_run)} biểu đồ cần tạo...")

    for metric_title, (train_key, val_key) in metrics_to_run.items():
        final_title = f"{metric_title} {base_title_suffix}"

        clean_name = metric_title.lower().replace('comparison', '').strip()
        clean_name = re.sub(r'\s+', '_', clean_name)  # thay khoảng trắng bằng gạch dưới
        final_filename = f"{base_filename_prefix}_{clean_name}"

        print(f"\n--- Đang vẽ: {metric_title} ---")

        create_comparison_plot(
            models_to_plot,
            final_filename,
            final_title,
            train_key,
            val_key,
            smooth_window=smooth_window_setting
        )

    print("\n✨ Hoàn thành tất cả biểu đồ.")