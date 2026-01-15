import pandas as pd
from pathlib import Path

# Liệt kê các thư mục model (mỗi thư mục có epoch_results.csv)
model_dirs = [
    r"DeepLabV3Plus/plot/plot_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0",
    r"PAN/plot/plot_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s0",
    r"MAnet/plot/plot_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s0",
    r"UPerNet/plot/plot_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s0",
    r"PSPNet/plot/plot_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s0",
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
    acc_col_1 = "val_accuracy"
    f1_col_1 = "val_f1"
    iou_col_1 = "val_iou"
    loss_col_1 = "val_loss"


    for c in [epoch_col, acc_col, f1_col, iou_col, loss_col, acc_col_1, f1_col_1, iou_col_1, loss_col_1]:
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
        "val_accuracy": float(row[acc_col_1]),
        "val_f1_score": float(row[f1_col_1]),
        "val_iou": float(row[iou_col_1]),
        "val_loss": float(row[loss_col_1]),
    })

summary_df = pd.DataFrame(summary_rows).sort_values("train_accuracy", ascending=False)
summary_df.to_csv("Massachusetts_s2.csv", index=False)
print("✅ Saved best_by_accuracy_all_models.csv")