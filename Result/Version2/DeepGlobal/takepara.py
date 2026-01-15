import pandas as pd
from pathlib import Path

# Liệt kê các thư mục model (mỗi thư mục có epoch_results.csv)
model_dirs = [
    r"DeepLabV3Plus/plot/plot_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1",
    r"PAN/plot/plot_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s1",
    r"MAnet/plot/plot_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s1",
    r"UPerNet/plot/plot_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s1",
    r"PSPNet/plot/plot_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s1",
]

summary_rows = []

for d in model_dirs:
    csv_path = Path(d) / "epoch_results.csv"
    if not csv_path.exists():
        print(f"⚠️ Không tìm thấy {csv_path}, bỏ qua.")
        continue

    df = pd.read_csv(csv_path)

    epoch_col = "epoch"
    acc_col   = "train_accuracy"
    f1_col    = "train_f1"
    iou_col   = "train_iou"
    loss_col  = "train_loss"

    for c in [epoch_col, acc_col, f1_col, iou_col]:
        if c not in df.columns:
            raise ValueError(f"Thiếu cột {c} trong {csv_path}")

    idx = df[acc_col].idxmax()
    row = df.loc[idx]

    summary_rows.append({
        "model": Path(d).name,
        "best_epoch_by_train_acc": int(row[epoch_col]),
        "train_accuracy": float(row[acc_col]),
        "train_f1": float(row[f1_col]),
        "train_iou": float(row[iou_col]),
        "train_loss": float(row[loss_col]),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("train_accuracy", ascending=False)
summary_df.to_csv("Massachusetts_s2.csv", index=False)
print("✅ Saved best_by_accuracy_all_models.csv")