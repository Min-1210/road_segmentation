import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_training_plots(history, config):
    plot_dir = config['training']['plot_dir']
    num_epochs = len(history['train_loss'])
    os.makedirs(plot_dir, exist_ok=True)
    loss_name = config['loss']['name']

    for key, value in history.items():
        np.save(os.path.join(plot_dir, f'{key}.npy'), np.array(value))
    print(f"\nLịch sử huấn luyện đã được lưu vào thư mục '{plot_dir}'")

    epochs_range = range(1, num_epochs + 1)
    plt.style.use('seaborn-v0_8-whitegrid')

    plot_pairs = {
        f'Loss ({loss_name})': ('train_loss', 'val_loss'),
        'IoU Score': ('train_iou_score', 'val_iou_score'),
        'F1-Score': ('train_f1_score', 'val_f1_score'),
        'Pixel Accuracy': ('train_accuracy', 'val_accuracy'),
        'Dice Loss': ('train_dice_loss', 'val_dice_loss'),
        'Focal Loss': ('train_focal_loss', 'val_focal_loss'),
    }

    num_plots = len(plot_pairs)
    num_cols = 3
    num_rows = math.ceil(num_plots / num_cols)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 5))
    fig.suptitle('Kết quả Huấn luyện Mô hình', fontsize=16, y=0.97)
    
    if num_rows == 1: axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, keys) in enumerate(plot_pairs.items()):
        ax = axes[i]
        x = list(epochs_range)
        
        if keys[0] in history:
            ax.plot(x, history[keys[0]], 'o-', label='Train')
        if keys[1] in history:
            ax.plot(x, history[keys[1]], 'o-', label='Validation')
            
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()

    for i in range(len(plot_pairs), len(axes)):
        axes[i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(plot_dir, "training_metrics_summary.png")
    plt.savefig(plot_path)
    print(f"Biểu đồ tổng hợp đã được lưu tại: {plot_path}")
    plt.close()

def save_confusion_matrix(stats, path, 
                          class_labels=('Predicted Road', 'Predicted Background'), 
                          true_labels=('Actual Road', 'Actual Background')):

    tp, fp, fn, tn = stats['tp'], stats['fp'], stats['fn'], stats['tn']
    total_tp = tp.item() if hasattr(tp, 'item') else tp
    total_fp = fp.item() if hasattr(fp, 'item') else fp
    total_fn = fn.item() if hasattr(fn, 'item') else fn
    total_tn = tn.item() if hasattr(tn, 'item') else tn
    
    matrix = np.array([[total_tp, total_fp], [total_fn, total_tn]])

    group_counts = [f"{value:,.0f}" for value in matrix.flatten()]
    total_pixels = matrix.sum()
    if total_pixels == 0: total_pixels = 1
    group_percentages = [f"{value / total_pixels:.2%}" for value in matrix.flatten()]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=labels, fmt='', cmap='Blues',
                xticklabels=list(class_labels),
                yticklabels=list(true_labels))
    plt.xlabel('Dự đoán (Prediction)')
    plt.ylabel('Thực tế (Ground Truth)')
    plt.title('Confusion Matrix (Best Epoch)', fontsize=14)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"📈 Confusion matrix saved at: {path}")