import os
import numpy as np
import matplotlib.pyplot as plt


def save_training_plots(history, config):
    plot_dir = config['training']['plot_dir']
    num_epochs = config['training']['num_epochs']
    os.makedirs(plot_dir, exist_ok=True)
    loss_name = config['loss']['name']

    for key, value in history.items():
        np.save(os.path.join(plot_dir, f'{key}.npy'), np.array(value))
    print(f"\nLịch sử huấn luyện đã được lưu vào thư mục '{plot_dir}'")

    epochs_range = range(1, num_epochs + 1)
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('Kết quả Huấn luyện Mô hình', fontsize=16, y=0.95)

    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs_range, history['train_loss'], 'o-', label=f'Train {loss_name}')
    ax1.plot(epochs_range, history['val_loss'], 'o-', label=f'Val {loss_name}')
    ax1.set_title(f'Training Loss ({loss_name})')
    ax1.set_xlabel('Epoch');
    ax1.set_ylabel('Loss');
    ax1.legend()

    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs_range, history['train_iou_score'], 'o-', label='Train IoU Score')
    ax2.plot(epochs_range, history['val_iou_score'], 'o-', label='Val IoU Score')
    ax2.set_title('IoU Score (Jaccard Index)')
    ax2.set_xlabel('Epoch');
    ax2.set_ylabel('Score');
    ax2.legend()

    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(epochs_range, history['train_f1_score'], 'o-', label='Train F1-Score')
    ax3.plot(epochs_range, history['val_f1_score'], 'o-', label='Val F1-Score')
    ax3.set_title('F1-Score')
    ax3.set_xlabel('Epoch');
    ax3.set_ylabel('Score');
    ax3.legend()

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(epochs_range, history['train_dice_loss'], 'o-', label='Train Dice Loss')
    ax4.plot(epochs_range, history['val_dice_loss'], 'o-', label='Val Dice Loss')
    ax4.set_title('Dice Loss')
    ax4.set_xlabel('Epoch');
    ax4.set_ylabel('Loss');
    ax4.legend()

    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(epochs_range, history['train_focal_loss'], 'o-', label='Train Focal Loss')
    ax5.plot(epochs_range, history['val_focal_loss'], 'o-', label='Val Focal Loss')
    ax5.set_title('Focal Loss')
    ax5.set_xlabel('Epoch');
    ax5.set_ylabel('Loss');
    ax5.legend()

    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(epochs_range, history['train_accuracy'], 'o-', label='Train Accuracy')
    ax6.plot(epochs_range, history['val_accuracy'], 'o-', label='Val Accuracy')
    ax6.set_title('Pixel Accuracy')
    ax6.set_xlabel('Epoch');
    ax6.set_ylabel('Accuracy');
    ax6.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(plot_dir, "training_metrics_summary.png")
    plt.savefig(plot_path)
    print(f"Biểu đồ tổng hợp đã được lưu tại: {plot_path}")
    # plt.show()