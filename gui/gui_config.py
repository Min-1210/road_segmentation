import os
WEIGHTS_DIR = "weights"
MODEL_DB = {
    "DeepLabV3Plus": {
        "mobileone_s0": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s2.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s2.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/DeepLabV3Plus/model/model_DeepGlobal_CrossEntropyLoss_DeepLabV3Plus_mobileone_s3.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/DeepLabV3Plus/model/model_Massachusetts_CrossEntropyLoss_DeepLabV3Plus_mobileone_s3.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/DeepLabV3Plus/model/model_TGRS_Road_CrossEntropyLoss_DeepLabV3Plus_mobileone_s3.pt"
        }     
    },
    "EfficientViT-Seg": {
        "efficientvit-seg-l1-ade20k": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l1-ade20k.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l1-ade20k.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l1-ade20k.pt"
        },
        "efficientvit-seg-l2-ade20k": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l2-ade20k.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l2-ade20k.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l2-ade20k.pt"
        },
        "efficientvit-seg-l1-cityscapes": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l1-cityscapes.pt"
        },
        "efficientvit-seg-l2-cityscapes": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/EfficientViT_Seg/model/model_DeepGlobal_CrossEntropyLoss_EfficientViT-Seg_l2-cityscapes.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/EfficientViT_Seg/model/model_Massachusetts_CrossEntropyLoss_EfficientViT-Seg_l2-cityscapes.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/EfficientViT_Seg/model/model_TGRS_Road_CrossEntropyLoss_EfficientViT-Seg_l2-cityscapes.pt"
        }
    },
    "FPN": {
        "mobileone_s0": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s0.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s0.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s1.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s1.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s2.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s2.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/FPN/model/model_DeepGlobal_CrossEntropyLoss_FPN_mobileone_s3.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/FPN/model/model_Massachusetts_CrossEntropyLoss_FPN_mobileone_s3.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/FPN/model/model_TGRS_Road_CrossEntropyLoss_FPN_mobileone_s3.pt"
        }     
    },
    "MAnet": {
        "mobileone_s0": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s0.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s0.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s1.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s1.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s2.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s2.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/MAnet/model/model_DeepGlobal_CrossEntropyLoss_MAnet_mobileone_s3.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/MAnet/model/model_Massachusetts_CrossEntropyLoss_MAnet_mobileone_s3.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/MAnet/model/model_TGRS_Road_CrossEntropyLoss_MAnet_mobileone_s3.pt"
        }     
    },
    "PAN": {
        "mobileone_s0": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s0.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s0.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s1.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s1.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s2.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s2.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PAN/model/model_DeepGlobal_CrossEntropyLoss_PAN_mobileone_s3.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PAN/model/model_Massachusetts_CrossEntropyLoss_PAN_mobileone_s3.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PAN/model/model_TGRS_Road_CrossEntropyLoss_PAN_mobileone_s3.pt"
        }     
    },
    "PSPNet": {
        "mobileone_s0": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s0.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s0.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s1.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s1.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s2.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s2.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/PSPNet/model/model_DeepGlobal_CrossEntropyLoss_PSPNet_mobileone_s3.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/PSPNet/model/model_Massachusetts_CrossEntropyLoss_PSPNet_mobileone_s3.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/PSPNet/model/model_TGRS_Road_CrossEntropyLoss_PSPNet_mobileone_s3.pt"
        }     
    },
    "UPerNet": {
        "mobileone_s0": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s0.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s0.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s0.pt"
        },
        "mobileone_s1": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s1.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s1.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s1.pt"
        },
        "mobileone_s2": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s2.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s2.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s2.pt"
        },
        "mobileone_s3": {
            "DeepGlobal": "D:/Lab/road_segmentation/Result/Version2/DeepGlobal/UPerNet/model/model_DeepGlobal_CrossEntropyLoss_UPerNet_mobileone_s3.pt",
            "Massachusetts": "D:/Lab/road_segmentation/Result/Version2/Massachusetts/UPerNet/model/model_Massachusetts_CrossEntropyLoss_UPerNet_mobileone_s3.pt",
            "TGRS_Road": "D:/Lab/road_segmentation/Result/Version2/TGRS_Road/UPerNet/model/model_TGRS_Road_CrossEntropyLoss_UPerNet_mobileone_s3.pt"
        }     
    },
}

def get_available_datasets(arch, encoder):
    if arch in MODEL_DB and encoder in MODEL_DB[arch]:
        return list(MODEL_DB[arch][encoder].keys())
    return []

def get_weight_path(arch, encoder, dataset):
    if arch in MODEL_DB and encoder in MODEL_DB[arch] and dataset in MODEL_DB[arch][encoder]:
        filename = MODEL_DB[arch][encoder][dataset]
        return os.path.abspath(os.path.join(WEIGHTS_DIR, filename))
    return None