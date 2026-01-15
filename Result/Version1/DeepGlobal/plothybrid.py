import os
import numpy as np
import matplotlib.pyplot as plt

def create_comparison_plot(models_to_plot, output_filename):
    metrics = {
        'Loss Comparison': ('train_loss', 'val_loss'),
        'IoU Score Comparison': ('train_iou_score', 'val_iou_score'),
        'F1-Score Comparison': ('train_f1_score', 'val_f1_score'),
        'Pixel Accuracy Comparison': ('train_accuracy', 'val_accuracy')
    }

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()  # Flatten the axes array for easy iteration
    fig.suptitle('Training Performance Comparison of Models (Resnet18)', fontsize=20, y=0.98)

    # Define colors and linestyles to differentiate models and train/val sets
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red
    linestyles = ['-', '--']  # Solid for Train, Dashed for Validation

    # Iterate over each metric to create a subplot
    for i, (title, (train_key, val_key)) in enumerate(metrics.items()):
        ax = axes[i]

        # Iterate over each model to draw its lines on the current subplot
        for j, model_info in enumerate(models_to_plot):
            model_name = model_info['name']
            model_path = model_info['path']
            
            # Construct the full paths to the .npy files
            train_history_path = os.path.join(model_path, f'{train_key}.npy')
            val_history_path = os.path.join(model_path, f'{val_key}.npy')

            # Check if the files exist before trying to load them
            if not os.path.exists(train_history_path) or not os.path.exists(val_history_path):
                # SỬA LỖI 1: Đưa emoji vào trong dấu ngoặc kép
                print(f"⚠️ Warning: .npy files not found for model '{model_name}' in '{model_path}'. Skipping this plot.")
                continue

            # Load the history data
            train_history = np.load(train_history_path)
            val_history = np.load(val_history_path)
            epochs = range(1, len(train_history) + 1)

            # Plot the train and validation lines
            ax.plot(epochs, train_history, linestyle=linestyles[0], color=colors[j], label=f"{model_name} (Train)")
            ax.plot(epochs, val_history, linestyle=linestyles[1], color=colors[j], label=f"{model_name} (Val)")

        # Set title and labels for each subplot
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.legend()
        ax.grid(True)

    # Adjust layout and save the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"✅ Comparison plot saved at: {output_filename}")


if __name__ == '__main__':
    # --- CONFIGURE MODELS TO PLOT ---
    # Change the "name" (for display) and "path" (to the plot folder) accordingly
    models_to_plot = [
        {
            "name": "DeepLabV3+",
            "path": r"plotDLV3+\plot_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_resnet18"
        },
        {
            "name": "Unet++",
            "path": r"plotUnet\plot_Massachusetts_CrossEntropyLoss_UNet++_resnet18"
        },
        {
            "name": "SegFormer",
            "path": r"plotSeg\plot_Massachusetts_CrossEntropyLoss_SegFormer_resnet18"
        }
    ]

    output_filename = "model_comparison_summary_resnet18.png"
    
    create_comparison_plot(models_to_plot, output_filename)