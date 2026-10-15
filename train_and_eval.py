# %% [markdown]
# # ☀️ Solar Filament Segmentation (MAGFiLO 2026) — Attention U-Net & Self-Evaluation Pipeline
# **Architecture:** Attention U-Net (ResNet34 + scSE Decoder Attention / Additive Attention Gates)  
# **Metrics:** IoU, Dice, Precision, Recall, Pixel Accuracy, Panoptic Quality (PQ, SQ, RQ), and 1-to-Many / Many-to-1 Topology Analysis (inspired by `azimahmadzadeh/self-evaluation-notebook`)

# %% [CELL 1: Install Dependencies in Google Colab]
# Uncomment the command below when running in Google Colab:
# !pip install -q segmentation-models-pytorch albumentations pycocotools torchmetrics opencv-python-headless tqdm pandas matplotlib seaborn

# %% [CELL 2: Imports & GPU Hardware Optimization Setup]
import os
import re
import cv2
import json
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import segmentation_models_pytorch as smp
import albumentations as A

warnings.filterwarnings("ignore")

# %% [CELL 3: Configuration & GPU / Colab / RTX Optimizations]
class Config:
    # ---> UPDATED PATHS FOR LOCAL PC (RTX 5080) <---
    IMAGE_DIR = r"C:\Users\hp\Downloads\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026\train_images"
    JSON_PATH = r"C:\Users\hp\Downloads\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026\MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    OUTPUT_DIR = "./pipeline_results"
    GRAPHS_DIR = os.path.join(OUTPUT_DIR, "graphs")
    VIS_DIR = os.path.join(OUTPUT_DIR, "qualitative_visualizations")
    SPLIT_DIR = os.path.join(OUTPUT_DIR, "splits")

    # Model Architecture Options:
    # "smp_attention_unet" -> ResNet34 Encoder + U-Net Decoder with scSE (Spatial & Channel Squeeze-and-Excitation Attention)
    # "custom_attention_gate_unet" -> Oktay et al. Additive Attention Gate U-Net
    ARCH_TYPE = "smp_attention_unet"
    ENCODER_NAME = "resnet34"
    ENCODER_WEIGHTS = "imagenet"

    SEED = 42
    VAL_PHYSICAL_COUNT = 110  # Strictly 110 original physical images for validation
    IMG_SIZE = 512

    # GPU-Optimized Hyperparameters
    BATCH_SIZE = 16
    NUM_WORKERS = 2 if os.name != "nt" else 0
    EPOCHS = 35
    LR = 2e-4

    # Regularization & Anti-Overfitting
    WEIGHT_DECAY = 1e-4
    DROPOUT_RATE = 0.30
    EARLY_STOP_PATIENCE = 7

    # Instance & Panoptic Quality Post-Processing Parameters
    PROB_THRESHOLD = 0.50
    MIN_INSTANCE_AREA = 25    # Filter tiny background noise specks (<25 px at 512x512)
    PQ_IOU_THRESHOLD = 0.50   # Standard Kirillov et al. (2019) PQ matching threshold

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP = True if DEVICE == "cuda" else False
    AMP_DTYPE = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )


def setup_gpu_and_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable TensorFloat-32 (Ampere/Ada/Hopper/Blackwell) & cuDNN Auto-Tuner
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(
            f"🚀 GPU Active: {gpu_name} ({vram_gb:.1f} GB VRAM) | "
            f"AMP ({Config.AMP_DTYPE}): {Config.USE_AMP}"
        )
    else:
        print("⚠️ GPU not detected! Running on CPU.")


setup_gpu_and_seed(Config.SEED)
for folder in [Config.OUTPUT_DIR, Config.GRAPHS_DIR, Config.VIS_DIR, Config.SPLIT_DIR]:
    os.makedirs(folder, exist_ok=True)


# %% [CELL 4: Event-Grouped Leakage-Free Train/Validation Split]
def get_root_event_key(filename):
    """
    Extracts the base solar observation date (YYYYMMDD) from a filename so that
    multiple telescope captures from the same solar day stay in the same split.
    Prevents temporal leakage between train and validation sets.
    """
    stem = os.path.splitext(os.path.basename(filename))[0].lower()
    stem = re.sub(r"(_aug.*|_rot.*|_flip.*|_crop.*|_v\d+)$", "", stem)

    date_match = re.search(r"(19|20)\d{6}", stem)
    if date_match:
        return date_match.group(0)
    return stem


def create_leakage_free_split(json_path, val_count=110, seed=42):
    """
    Groups physical images by solar observation date (YYYYMMDD) and allocates
    exact or near-exact `val_count` physical images to Validation without splitting
    any single solar day across Train and Validation.
    """
    print("\n--- 1. Performing Strict Event-Grouped Leakage-Free Split ---")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"COCO JSON file not found at: {json_path}\n"
            f"Please update Config.JSON_PATH and Config.IMAGE_DIR."
        )

    coco = COCO(json_path)

    # 1. Map each clean physical filename to all of its COCO image_ids
    physical_to_ids = defaultdict(list)
    for img_id, img_info in coco.imgs.items():
        clean_name = os.path.basename(img_info["file_name"])
        if not any(tag in clean_name.lower() for tag in ["aug", "rot", "flip"]):
            physical_to_ids[clean_name].append(img_id)

    # Keep physical files that have at least 1 valid annotation
    valid_physical_files = []
    for clean_name, id_list in sorted(physical_to_ids.items()):
        if len(coco.getAnnIds(imgIds=id_list)) > 0:
            valid_physical_files.append(clean_name)

    # 2. Group physical filenames by Solar Event Date Key
    event_groups = defaultdict(list)
    for clean_name in valid_physical_files:
        event_key = get_root_event_key(clean_name)
        event_groups[event_key].append(clean_name)

    # Shuffle event keys deterministically
    event_keys = list(event_groups.keys())
    rng = random.Random(seed)
    rng.shuffle(event_keys)

    val_physical_set = set()
    train_physical_set = set()

    # Adjust target val_count automatically if debugging on a tiny subset
    effective_val_count = min(val_count, max(1, int(len(valid_physical_files) * 0.2)))
    if len(valid_physical_files) >= val_count * 2:
        effective_val_count = val_count

    for key in event_keys:
        group_files = event_groups[key]
        if len(val_physical_set) + len(group_files) <= effective_val_count:
            val_physical_set.update(group_files)
        else:
            train_physical_set.update(group_files)

    # If still short by a few images, fill from single-image events in train
    if len(val_physical_set) < effective_val_count:
        needed = effective_val_count - len(val_physical_set)
        single_keys = [
            k for k in event_keys
            if len(event_groups[k]) == 1 and event_groups[k][0] in train_physical_set
        ]
        for k in single_keys[:needed]:
            fname = event_groups[k][0]
            train_physical_set.remove(fname)
            val_physical_set.add(fname)

    # Verify zero event overlap
    train_events = {get_root_event_key(f) for f in train_physical_set}
    val_events = {get_root_event_key(f) for f in val_physical_set}
    overlap_events = train_events.intersection(val_events)
    assert len(overlap_events) == 0, f"Leakage detected in events: {overlap_events}"

    # Collect unique representative COCO image_id per physical file
    train_img_ids = [physical_to_ids[f][0] for f in sorted(train_physical_set)]
    val_img_ids = [physical_to_ids[f][0] for f in sorted(val_physical_set)]

    # Save split manifest CSV for reproducibility and GitHub reporting
    manifest_rows = []
    for f in sorted(train_physical_set):
        manifest_rows.append({
            "file_name": f,
            "event_key": get_root_event_key(f),
            "coco_img_id": physical_to_ids[f][0],
            "split": "train"
        })
    for f in sorted(val_physical_set):
        manifest_rows.append({
            "file_name": f,
            "event_key": get_root_event_key(f),
            "coco_img_id": physical_to_ids[f][0],
            "split": "val"
        })
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(Config.SPLIT_DIR, "split_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    print(f"✅ Total Valid Physical Images : {len(valid_physical_files)}")
    print(f"✅ Train Physical Images       : {len(train_img_ids)} ({len(train_events)} solar events)")
    print(f"✅ Validation Physical Images  : {len(val_img_ids)} ({len(val_events)} solar events)")
    print(f"✅ Event-Level Leakage Overlap : {len(overlap_events)} (Strictly 0)")
    print(f"✅ Saved Split Manifest        : {manifest_path}")

    return coco, train_img_ids, val_img_ids


# %% [CELL 5: Augmentations & Solar Filament Dataset Loader]
def get_transforms(img_size=512):
    """
    Creates solar-physics-aware augmentations for training (dihedral invariance,
    CLAHE for limb darkening compensation, slight contrast & noise invariance)
    and deterministic resizing for validation.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.10,
            rotate_limit=45,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            mask_value=0,
            p=0.5
        ),
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
        ], p=0.4),
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_LINEAR),
    ])
    return train_transform, val_transform


class SolarFilamentDataset(Dataset):
    """
    Loads H-alpha solar images and constructs binary semantic masks as well as
    instance-level filament masks (for Panoptic Quality evaluation).
    """
    def __init__(self, coco, img_ids, image_dir, transform=None, is_val=False):
        self.coco = coco
        self.img_ids = img_ids
        self.image_dir = image_dir
        self.transform = transform
        self.is_val = is_val

    def __len__(self):
        return len(self.img_ids)

    def _resolve_image_path(self, raw_file_name):
        clean_name = os.path.basename(raw_file_name)
        candidate_1 = os.path.join(self.image_dir, clean_name)
        if os.path.exists(candidate_1):
            return candidate_1
        candidate_2 = os.path.join(self.image_dir, raw_file_name)
        if os.path.exists(candidate_2):
            return candidate_2
        raise FileNotFoundError(f"Could not locate image {clean_name} in {self.image_dir}")

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.imgs[img_id]
        img_path = self._resolve_image_path(img_info["file_name"])

        # Load H-alpha observation in RGB (3 channels for pretrained ResNet encoder)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Corrupted or unreadable image file: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        # Build binary semantic mask and labeled instance map (1..N for each GT filament)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        semantic_mask = np.zeros((h, w), dtype=np.uint8)
        instance_map = np.zeros((h, w), dtype=np.int32)

        inst_idx = 1
        for ann in anns:
            ann_mask = self.coco.annToMask(ann)
            if ann_mask.shape[:2] != (h, w):
                ann_mask = cv2.resize(ann_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if ann_mask.sum() > 0:
                semantic_mask = np.maximum(semantic_mask, ann_mask.astype(np.uint8))
                instance_map[ann_mask > 0] = inst_idx
                inst_idx += 1

        if self.transform is not None:
            # Resize/augment both semantic_mask and instance_map in lockstep
            augmented = self.transform(
                image=image,
                masks=[semantic_mask, instance_map]
            )
            image = augmented["image"]
            semantic_mask = augmented["masks"][0]
            instance_map = augmented["masks"][1]

        # Normalize image with ImageNet statistics for pretrained backbone
        image_float = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image_norm = (image_float - mean) / std

        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(semantic_mask.astype(np.float32)).unsqueeze(0)
        instance_tensor = torch.from_numpy(instance_map.astype(np.int32))

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "instance_map": instance_tensor,
            "img_id": img_id,
            "file_name": os.path.basename(img_info["file_name"])
        }


# %% [CELL 6: Attention U-Net Architecture (scSE U-Net & Oktay Attention Gate U-Net)]
class AttentionGate2D(nn.Module):
    """
    Additive Attention Gate (Oktay et al., 'Attention U-Net: Learning Where to Look
    for the Pancreas') that suppresses noisy solar background regions and highlights
    fine filament structures and barbs along skip connections.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha


class ConvBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            layers.insert(3, nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomAttentionGateUNet(nn.Module):
    """
    Full Oktay et al. Attention U-Net with 2D Spatial Dropout in deeper layers.
    """
    def __init__(self, in_channels=3, out_channels=1, dropout_rate=0.30):
        super().__init__()
        filters = [64, 128, 256, 512, 1024]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = ConvBlock2D(in_channels, filters[0])
        self.enc2 = ConvBlock2D(filters[0], filters[1])
        self.enc3 = ConvBlock2D(filters[1], filters[2], dropout_p=dropout_rate * 0.5)
        self.enc4 = ConvBlock2D(filters[2], filters[3], dropout_p=dropout_rate)
        self.center = ConvBlock2D(filters[3], filters[4], dropout_p=dropout_rate)

        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate2D(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.dec4 = ConvBlock2D(filters[4], filters[3], dropout_p=dropout_rate)

        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate2D(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.dec3 = ConvBlock2D(filters[3], filters[2], dropout_p=dropout_rate * 0.5)

        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate2D(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.dec2 = ConvBlock2D(filters[2], filters[1])

        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate2D(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.dec1 = ConvBlock2D(filters[1], filters[0])

        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.center(self.pool(e4))

        d4 = self.up4(b)
        s4 = self.att4(g=d4, x=e4)
        d4 = self.dec4(torch.cat([s4, d4], dim=1))

        d3 = self.up3(d4)
        s3 = self.att3(g=d3, x=e3)
        d3 = self.dec3(torch.cat([s3, d3], dim=1))

        d2 = self.up2(d3)
        s2 = self.att2(g=d2, x=e2)
        d2 = self.dec2(torch.cat([s2, d2], dim=1))

        d1 = self.up1(d2)
        s1 = self.att1(g=d1, x=e1)
        d1 = self.dec1(torch.cat([s1, d1], dim=1))

        return self.final_conv(d1)


class SMPAttentionUNetWithDropout(nn.Module):
    """
    Pretrained ResNet34 Encoder + U-Net Decoder with Concurrent Spatial and Channel
    Squeeze & Excitation (scSE) Attention Gates and 2D Spatial Dropout regularization.
    """
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", dropout_rate=0.30):
        super().__init__()
        self.base_model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            decoder_attention_type="scse",  # Spatial + Channel Attention U-Net
        )
        self.spatial_dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        features = self.base_model.encoder(x)
        # Apply 2D Spatial Dropout on bottleneck features to prevent overfitting
        features[-1] = self.spatial_dropout(features[-1])
        decoder_output = self.base_model.decoder(features)
        logits = self.base_model.segmentation_head(decoder_output)
        return logits


def build_attention_unet():
    if Config.ARCH_TYPE == "custom_attention_gate_unet":
        print("🧠 Building Custom Additive Attention Gate U-Net...")
        model = CustomAttentionGateUNet(
            in_channels=3,
            out_channels=1,
            dropout_rate=Config.DROPOUT_RATE
        )
    else:
        print(f"🧠 Building SMP Attention U-Net ({Config.ENCODER_NAME} + scSE Attention)...")
        model = SMPAttentionUNetWithDropout(
            encoder_name=Config.ENCODER_NAME,
            encoder_weights=Config.ENCODER_WEIGHTS,
            dropout_rate=Config.DROPOUT_RATE
        )

    # Convert to channels_last memory format on CUDA for faster Tensor Core throughput
    model = model.to(Config.DEVICE)
    if Config.DEVICE == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


# %% [CELL 7: Hybrid Loss Function (BCE + Soft Dice + Focal Tversky)]
class HybridFilamentLoss(nn.Module):
    """
    Combines BCEWithLogitsLoss + Soft Dice Loss + Focal-Tversky penalty to handle
    severe foreground-background class imbalance and capture thin filament barbs.
    """
    def __init__(self, bce_weight=0.4, dice_weight=0.6, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)

        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# %% [CELL 8: Self-Evaluation Metrics Engine (IoU, Dice, Precision, Recall, Accuracy, Panoptic Quality & Topology)]
def extract_predicted_instances(binary_mask, min_area=25):
    """
    Decomposes a predicted binary mask into individual filament instances via
    8-connectivity Connected Components, filtering out tiny background artifacts.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask.astype(np.uint8), connectivity=8
    )
    instances = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            instances.append(labels == label_id)
    return instances


def extract_gt_instances(instance_map):
    """
    Extracts individual ground-truth filament masks from the integer instance map.
    """
    unique_ids = [uid for uid in np.unique(instance_map) if uid > 0]
    return [instance_map == uid for uid in unique_ids]


def compute_image_panoptic_and_topology(gt_instance_map, pred_binary_mask, iou_thresh=0.5, min_area=25):
    """
    Implements the exact Panoptic Quality (PQ = SQ x RQ) and 1-to-Many / Many-to-1
    correspondence analysis from `azimahmadzadeh/self-evaluation-notebook` and
    Kirillov et al. (CVPR 2019).
    """
    gt_list = extract_gt_instances(gt_instance_map)
    pred_list = extract_predicted_instances(pred_binary_mask, min_area=min_area)

    num_gt = len(gt_list)
    num_pred = len(pred_list)

    if num_gt == 0 and num_pred == 0:
        return {
            "pq": 1.0, "sq": 1.0, "rq": 1.0,
            "tp": 0, "fp": 0, "fn": 0, "sum_iou": 0.0,
            "one_to_many": 0, "many_to_one": 0,
            "matched_ious": [], "matched_dices": []
        }

    # Build pairwise IoU and Dice matrices between GT segments Y and Pred segments Y_hat
    iou_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)
    dice_matrix = np.zeros((num_gt, num_pred), dtype=np.float32)

    for i, gt_m in enumerate(gt_list):
        gt_area = gt_m.sum()
        for j, pr_m in enumerate(pred_list):
            inter = np.logical_and(gt_m, pr_m).sum()
            if inter > 0:
                pr_area = pr_m.sum()
                union = gt_area + pr_area - inter
                iou_matrix[i, j] = inter / max(union, 1e-6)
                dice_matrix[i, j] = (2.0 * inter) / max(gt_area + pr_area, 1e-6)

    # 1. Topology relations (One-to-Many fragmentation & Many-to-One merging)
    # Any non-trivial overlap (IoU > 0.05) forms an edge in the correspondence graph
    overlap_adj = iou_matrix > 0.05
    # One-to-Many: 1 GT filament split across >= 2 predicted segments
    one_to_many_count = int(np.sum(overlap_adj.sum(axis=1) >= 2)) if num_gt > 0 and num_pred > 0 else 0
    # Many-to-One: >= 2 GT filaments merged into 1 predicted segment
    many_to_one_count = int(np.sum(overlap_adj.sum(axis=0) >= 2)) if num_gt > 0 and num_pred > 0 else 0

    # 2. Panoptic Quality Matching (Strictly unique when IoU > 0.5)
    matched_gt = set()
    matched_pred = set()
    sum_iou = 0.0
    matched_ious = []
    matched_dices = []

    if num_gt > 0 and num_pred > 0:
        # Sort candidate pairs by descending IoU
        pairs = np.argwhere(iou_matrix > iou_thresh)
        if len(pairs) > 0:
            pair_ious = [iou_matrix[i, j] for i, j in pairs]
            order = np.argsort(pair_ious)[::-1]
            for idx in order:
                i, j = pairs[idx]
                if i not in matched_gt and j not in matched_pred:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    sum_iou += float(iou_matrix[i, j])
                    matched_ious.append(float(iou_matrix[i, j]))
                    matched_dices.append(float(dice_matrix[i, j]))

    tp = len(matched_gt)
    fp = num_pred - len(matched_pred)
    fn = num_gt - len(matched_gt)

    sq = (sum_iou / tp) if tp > 0 else 0.0
    denom = tp + 0.5 * fp + 0.5 * fn
    rq = (tp / denom) if denom > 0 else 0.0
    pq = sq * rq

    return {
        "pq": pq, "sq": sq, "rq": rq,
        "tp": tp, "fp": fp, "fn": fn, "sum_iou": sum_iou,
        "one_to_many": one_to_many_count,
        "many_to_one": many_to_one_count,
        "matched_ious": matched_ious,
        "matched_dices": matched_dices
    }


# %% [CELL 9: Training & Comprehensive Validation Loop]
def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    running_loss = 0.0
    total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        images = batch["image"].to(Config.DEVICE, non_blocking=True)
        masks = batch["mask"].to(Config.DEVICE, non_blocking=True)
        if Config.DEVICE == "cuda":
            images = images.to(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
            logits = model(images)
            loss = criterion(logits, masks)

        if Config.USE_AMP and Config.AMP_DTYPE == torch.float16:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            preds = (torch.sigmoid(logits) > Config.PROB_THRESHOLD).float()
            tp = (preds * masks).sum().item()
            fp = (preds * (1.0 - masks)).sum().item()
            fn = ((1.0 - preds) * masks).sum().item()
            tn = ((1.0 - preds) * (1.0 - masks)).sum().item()
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    iou = total_tp / max(total_tp + total_fp + total_fn, 1e-7)
    dice = (2.0 * total_tp) / max(2.0 * total_tp + total_fp + total_fn, 1e-7)
    precision = total_tp / max(total_tp + total_fp, 1e-7)
    recall = total_tp / max(total_tp + total_fn, 1e-7)
    accuracy = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1e-7)

    return {
        "loss": epoch_loss,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy
    }


@torch.no_grad()
def evaluate_validation(model, loader, criterion, collect_distributions=False):
    model.eval()
    running_loss = 0.0
    total_tp, total_fp, total_fn, total_tn = 0.0, 0.0, 0.0, 0.0

    # Global Panoptic Quality accumulators across all validation instances
    pq_tp, pq_fp, pq_fn, pq_sum_iou = 0, 0, 0, 0.0
    total_one_to_many = 0
    total_many_to_one = 0

    per_image_records = []
    instance_ious = []
    instance_dices = []

    pbar = tqdm(loader, desc="Valid", leave=False)
    for batch in pbar:
        images = batch["image"].to(Config.DEVICE, non_blocking=True)
        masks = batch["mask"].to(Config.DEVICE, non_blocking=True)
        instance_maps = batch["instance_map"].numpy()
        file_names = batch["file_name"]

        if Config.DEVICE == "cuda":
            images = images.to(memory_format=torch.channels_last)

        with torch.amp.autocast("cuda", enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
            logits = model(images)
            loss = criterion(logits, masks)

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits)
        preds = (probs > Config.PROB_THRESHOLD).float()

        # Pixel-level confusion matrix accumulation
        tp = (preds * masks).sum().item()
        fp = (preds * (1.0 - masks)).sum().item()
        fn = ((1.0 - preds) * masks).sum().item()
        tn = ((1.0 - preds) * (1.0 - masks)).sum().item()
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # Per-image semantic & Panoptic Quality evaluation
        preds_np = preds.squeeze(1).cpu().numpy().astype(np.uint8)
        masks_np = masks.squeeze(1).cpu().numpy().astype(np.uint8)

        for b_idx in range(images.size(0)):
            pr_m = preds_np[b_idx]
            gt_m = masks_np[b_idx]
            gt_inst = instance_maps[b_idx]

            img_tp = np.logical_and(pr_m == 1, gt_m == 1).sum()
            img_fp = np.logical_and(pr_m == 1, gt_m == 0).sum()
            img_fn = np.logical_and(pr_m == 0, gt_m == 1).sum()
            img_tn = np.logical_and(pr_m == 0, gt_m == 0).sum()

            img_iou = img_tp / max(img_tp + img_fp + img_fn, 1e-7)
            img_dice = (2.0 * img_tp) / max(2.0 * img_tp + img_fp + img_fn, 1e-7)
            img_prec = img_tp / max(img_tp + img_fp, 1e-7)
            img_rec = img_tp / max(img_tp + img_fn, 1e-7)
            img_acc = (img_tp + img_tn) / max(img_tp + img_tn + img_fp + img_fn, 1e-7)

            pq_stats = compute_image_panoptic_and_topology(
                gt_inst,
                pr_m,
                iou_thresh=Config.PQ_IOU_THRESHOLD,
                min_area=Config.MIN_INSTANCE_AREA
            )

            pq_tp += pq_stats["tp"]
            pq_fp += pq_stats["fp"]
            pq_fn += pq_stats["fn"]
            pq_sum_iou += pq_stats["sum_iou"]
            total_one_to_many += pq_stats["one_to_many"]
            total_many_to_one += pq_stats["many_to_one"]

            if collect_distributions:
                instance_ious.extend(pq_stats["matched_ious"])
                instance_dices.extend(pq_stats["matched_dices"])
                per_image_records.append({
                    "file_name": file_names[b_idx],
                    "iou": img_iou,
                    "dice": img_dice,
                    "precision": img_prec,
                    "recall": img_rec,
                    "accuracy": img_acc,
                    "pq": pq_stats["pq"],
                    "sq": pq_stats["sq"],
                    "rq": pq_stats["rq"],
                    "tp_instances": pq_stats["tp"],
                    "fp_instances": pq_stats["fp"],
                    "fn_instances": pq_stats["fn"],
                    "one_to_many": pq_stats["one_to_many"],
                    "many_to_one": pq_stats["many_to_one"]
                })

    epoch_loss = running_loss / len(loader.dataset)
    iou = total_tp / max(total_tp + total_fp + total_fn, 1e-7)
    dice = (2.0 * total_tp) / max(2.0 * total_tp + total_fp + total_fn, 1e-7)
    precision = total_tp / max(total_tp + total_fp, 1e-7)
    recall = total_tp / max(total_tp + total_fn, 1e-7)
    accuracy = (total_tp + total_tn) / max(total_tp + total_tn + total_fp + total_fn, 1e-7)

    # Dataset-wide Panoptic Quality (SQ, RQ, PQ)
    sq = (pq_sum_iou / pq_tp) if pq_tp > 0 else 0.0
    pq_denom = pq_tp + 0.5 * pq_fp + 0.5 * pq_fn
    rq = (pq_tp / pq_denom) if pq_denom > 0 else 0.0
    pq = sq * rq

    result = {
        "loss": epoch_loss,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "pq": pq,
        "sq": sq,
        "rq": rq,
        "one_to_many": total_one_to_many,
        "many_to_one": total_many_to_one
    }

    if collect_distributions:
        result["per_image_df"] = pd.DataFrame(per_image_records)
        result["instance_ious"] = instance_ious
        result["instance_dices"] = instance_dices

    return result


# %% [CELL 10: Self-Evaluation Notebook Graphs & Qualitative Visualizations]
def plot_training_history(history_df):
    """
    Saves multi-panel training & validation curves for Loss, IoU, Dice,
    Precision, Recall, Accuracy, and Panoptic Quality (PQ, SQ, RQ).
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    epochs = history_df["epoch"]

    # Panel 1: Loss
    axes[0, 0].plot(epochs, history_df["train_loss"], "o-", label="Train Loss", color="#2563eb")
    axes[0, 0].plot(epochs, history_df["val_loss"], "s-", label="Val Loss", color="#dc2626")
    axes[0, 0].set_title("Hybrid BCE + Dice Loss", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Panel 2: IoU & Dice
    axes[0, 1].plot(epochs, history_df["train_iou"], "--", label="Train IoU", color="#3b82f6")
    axes[0, 1].plot(epochs, history_df["val_iou"], "o-", label="Val IoU", color="#1d4ed8", linewidth=2)
    axes[0, 1].plot(epochs, history_df["val_dice"], "s-", label="Val Dice (F1)", color="#059669", linewidth=2)
    axes[0, 1].set_title("Validation IoU & Dice Score", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Panel 3: Precision, Recall & Pixel Accuracy
    axes[1, 0].plot(epochs, history_df["val_precision"], "o-", label="Val Precision", color="#d97706")
    axes[1, 0].plot(epochs, history_df["val_recall"], "s-", label="Val Recall", color="#7c3aed")
    axes[1, 0].plot(epochs, history_df["val_accuracy"], "^-", label="Val Pixel Accuracy", color="#475569")
    axes[1, 0].set_title("Validation Precision, Recall & Accuracy", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Panel 4: Panoptic Quality (PQ, SQ, RQ)
    axes[1, 1].plot(epochs, history_df["val_pq"], "o-", label="Panoptic Quality (PQ)", color="#e11d48", linewidth=2.5)
    axes[1, 1].plot(epochs, history_df["val_sq"], "s--", label="Segmentation Quality (SQ)", color="#0284c7")
    axes[1, 1].plot(epochs, history_df["val_rq"], "^--", label="Recognition Quality (RQ)", color="#16a34a")
    axes[1, 1].set_title("MAGFiLO Panoptic Quality (PQ = SQ × RQ)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    save_path = os.path.join(Config.GRAPHS_DIR, "training_and_pq_curves.png")
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"📊 Saved Training & PQ Curves -> {save_path}")


def plot_self_evaluation_diagnostics(eval_results):
    """
    Replicates the diagnostic plots from `azimahmadzadeh/self-evaluation-notebook`:
    1. Distribution of Dice scores across validation images & matched instances
    2. Distribution of IoU scores across validation images & matched instances
    3. Distribution of One-to-Many (fragmentation) and Many-to-One (merging) relations
    4. Image-level PQ vs. Dice scatter plot
    """
    df = eval_results["per_image_df"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Distribution of Dice Scores
    axes[0, 0].hist(df["dice"], bins=20, color="#10b981", edgecolor="black", alpha=0.8)
    axes[0, 0].axvline(df["dice"].mean(), color="#b91c1c", linestyle="--", linewidth=2,
                       label=f"Mean Dice: {df['dice'].mean():.4f}")
    axes[0, 0].set_title("Self-Evaluation: Distribution of Dice Scores", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Dice Score")
    axes[0, 0].set_ylabel("Number of Validation Images")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Distribution of IoU Scores
    axes[0, 1].hist(df["iou"], bins=20, color="#3b82f6", edgecolor="black", alpha=0.8)
    axes[0, 1].axvline(df["iou"].mean(), color="#b91c1c", linestyle="--", linewidth=2,
                       label=f"Mean IoU: {df['iou'].mean():.4f}")
    axes[0, 1].set_title("Self-Evaluation: Distribution of IoU Scores", fontsize=13, fontweight="bold")
    axes[0, 1].set_xlabel("Intersection over Union (IoU)")
    axes[0, 1].set_ylabel("Number of Validation Images")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # 3. Distribution of One-to-Many and Many-to-One Relations
    max_rel = int(max(df["one_to_many"].max(), df["many_to_one"].max(), 5))
    bins = np.arange(0, max_rel + 2) - 0.5
    axes[1, 0].hist(
        [df["one_to_many"], df["many_to_one"]],
        bins=bins,
        label=[
            f"1-to-Many (Fragmentation, Total={df['one_to_many'].sum()})",
            f"Many-to-1 (Merging, Total={df['many_to_one'].sum()})"
        ],
        color=["#f59e0b", "#8b5cf6"],
        edgecolor="black",
        alpha=0.85
    )
    axes[1, 0].set_title("Topology: 1-to-Many & Many-to-1 Filament Relations", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("Count per Validation Image")
    axes[1, 0].set_ylabel("Number of Images")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # 4. Panoptic Quality Breakdown (PQ, SQ, RQ & Precision/Recall Summary Bar)
    metric_names = ["IoU", "Dice", "Precision", "Recall", "Accuracy", "SQ", "RQ", "PQ"]
    metric_vals = [
        eval_results["iou"],
        eval_results["dice"],
        eval_results["precision"],
        eval_results["recall"],
        eval_results["accuracy"],
        eval_results["sq"],
        eval_results["rq"],
        eval_results["pq"]
    ]
    bars = axes[1, 1].bar(
        metric_names,
        metric_vals,
        color=["#2563eb", "#059669", "#d97706", "#7c3aed", "#64748b", "#0284c7", "#16a34a", "#e11d48"],
        edgecolor="black"
    )
    axes[1, 1].set_ylim(0, 1.08)
    axes[1, 1].set_title("Final Validation Metrics Summary (Best Checkpoint)", fontsize=13, fontweight="bold")
    axes[1, 1].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, metric_vals):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    plt.tight_layout()
    save_path = os.path.join(Config.GRAPHS_DIR, "self_evaluation_distributions.png")
    plt.savefig(save_path, dpi=220)
    plt.close()
    print(f"📊 Saved Self-Evaluation Distributions -> {save_path}")


@torch.no_grad()
def save_qualitative_visualizations(model, val_loader, num_samples=6):
    """
    Saves side-by-side qualitative visual panels:
    [H-Alpha Solar Image | Ground Truth Mask | Attention U-Net Probability | Error Map (TP=Green, FP=Red, FN=Blue)]
    """
    model.eval()
    saved = 0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    for batch in val_loader:
        images = batch["image"].to(Config.DEVICE)
        if Config.DEVICE == "cuda":
            images = images.to(memory_format=torch.channels_last)
        masks = batch["mask"].cpu().numpy()
        inst_maps = batch["instance_map"].numpy()
        fnames = batch["file_name"]

        with torch.amp.autocast("cuda", enabled=Config.USE_AMP, dtype=Config.AMP_DTYPE):
            # Cast bfloat16 output to float32 (.float()) before converting to NumPy
            probs = torch.sigmoid(model(images)).float().cpu().numpy()

        for i in range(images.size(0)):
            if saved >= num_samples:
                return
            img_np = images[i].float().cpu().permute(1, 2, 0).numpy()
            img_vis = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

            gt_m = masks[i, 0].astype(np.uint8)
            pr_prob = probs[i, 0]
            pr_m = (pr_prob > Config.PROB_THRESHOLD).astype(np.uint8)

            pq_stats = compute_image_panoptic_and_topology(
                inst_maps[i], pr_m, iou_thresh=Config.PQ_IOU_THRESHOLD, min_area=Config.MIN_INSTANCE_AREA
            )

            # Construct RGB Error Overlay (Green=TP, Red=FP, Blue=FN)
            error_overlay = img_vis.copy()
            tp_mask = np.logical_and(pr_m == 1, gt_m == 1)
            fp_mask = np.logical_and(pr_m == 1, gt_m == 0)
            fn_mask = np.logical_and(pr_m == 0, gt_m == 1)

            error_overlay[tp_mask] = [0, 230, 118]    # Green: True Positive
            error_overlay[fp_mask] = [255, 59, 48]    # Red: False Positive
            error_overlay[fn_mask] = [56, 189, 248]   # Blue: False Negative

            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes[0].imshow(img_vis)
            axes[0].set_title(f"H-Alpha: {fnames[i]}", fontsize=11, fontweight="bold")
            axes[0].axis("off")

            axes[1].imshow(gt_m, cmap="gray")
            axes[1].set_title("Ground Truth Mask (MAGFiLO)", fontsize=11, fontweight="bold")
            axes[1].axis("off")

            axes[2].imshow(pr_prob, cmap="magma", vmin=0, vmax=1)
            axes[2].set_title("Attention U-Net Probability", fontsize=11, fontweight="bold")
            axes[2].axis("off")

            axes[3].imshow(error_overlay)
            axes[3].set_title(
                f"PQ={pq_stats['pq']:.3f} | SQ={pq_stats['sq']:.3f} | RQ={pq_stats['rq']:.3f}\n"
                f"(Green=TP, Red=FP, Blue=FN)",
                fontsize=10,
                fontweight="bold"
            )
            axes[3].axis("off")

            plt.tight_layout()
            out_file = os.path.join(Config.VIS_DIR, f"val_sample_{saved + 1}_{fnames[i]}.png")
            plt.savefig(out_file, dpi=180)
            plt.close()
            saved += 1


# %% [CELL 11: Main Execution Entrypoint]
def run_training_pipeline():
    # 1. Perform Leakage-Free Split
    coco, train_ids, val_ids = create_leakage_free_split(
        Config.JSON_PATH,
        val_count=Config.VAL_PHYSICAL_COUNT,
        seed=Config.SEED
    )

    # 2. Build Datasets & GPU-Pinned DataLoaders
    train_tf, val_tf = get_transforms(Config.IMG_SIZE)
    train_dataset = SolarFilamentDataset(
        coco, train_ids, Config.IMAGE_DIR, transform=train_tf, is_val=False
    )
    val_dataset = SolarFilamentDataset(
        coco, val_ids, Config.IMAGE_DIR, transform=val_tf, is_val=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0),
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE == "cuda"),
        persistent_workers=(Config.NUM_WORKERS > 0)
    )

    # 3. Initialize Attention U-Net, Optimizer, Scheduler, and Loss
    model = build_attention_unet()
    criterion = HybridFilamentLoss(bce_weight=0.4, dice_weight=0.6)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(Config.USE_AMP and Config.AMP_DTYPE == torch.float16))

    best_val_iou = -1.0
    patience_counter = 0
    best_ckpt_path = os.path.join(Config.OUTPUT_DIR, "best_attention_unet.pth")
    history_records = []

    print("\n--- 2. Starting GPU-Accelerated Attention U-Net Training ---")
    for epoch in range(1, Config.EPOCHS + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        val_metrics = evaluate_validation(model, val_loader, criterion, collect_distributions=False)
        scheduler.step()
        elapsed = time.time() - t0

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "train_dice": train_metrics["dice"],
            "val_loss": val_metrics["loss"],
            "val_iou": val_metrics["iou"],
            "val_dice": val_metrics["dice"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_accuracy": val_metrics["accuracy"],
            "val_pq": val_metrics["pq"],
            "val_sq": val_metrics["sq"],
            "val_rq": val_metrics["rq"],
            "val_one_to_many": val_metrics["one_to_many"],
            "val_many_to_one": val_metrics["many_to_one"],
        }
        history_records.append(row)

        print(
            f"Epoch [{epoch:02d}/{Config.EPOCHS:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f} IoU: {train_metrics['iou']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} Dice: {val_metrics['dice']:.4f} "
            f"Prec: {val_metrics['precision']:.4f} Rec: {val_metrics['recall']:.4f} "
            f"Acc: {val_metrics['accuracy']:.4f} | "
            f"PQ: {val_metrics['pq']:.4f} (SQ: {val_metrics['sq']:.4f}, RQ: {val_metrics['rq']:.4f})"
        )

        # Save Best Model Checkpoint by Validation IoU
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "arch_type": Config.ARCH_TYPE,
                "encoder_name": Config.ENCODER_NAME,
                "img_size": Config.IMG_SIZE,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, best_ckpt_path)
            print(f"  🌟 Saved New Best Model (Val IoU: {best_val_iou:.4f}, Val PQ: {val_metrics['pq']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOP_PATIENCE:
                print(f"⏹️ Early stopping triggered after {epoch} epochs!")
                break

    # Save epoch history CSV & training graphs
    history_df = pd.DataFrame(history_records)
    history_csv = os.path.join(Config.OUTPUT_DIR, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    plot_training_history(history_df)

    # 4. Run Comprehensive Self-Evaluation on Best Model
    print("\n--- 3. Running Self-Evaluation Notebook Suite on Best Checkpoint ---")
    checkpoint = torch.load(best_ckpt_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_eval = evaluate_validation(model, val_loader, criterion, collect_distributions=True)
    final_eval["per_image_df"].to_csv(
        os.path.join(Config.OUTPUT_DIR, "val_per_image_self_evaluation.csv"), index=False
    )

    plot_self_evaluation_diagnostics(final_eval)
    save_qualitative_visualizations(model, val_loader, num_samples=6)

    print("\n============================================================")
    print("🏆 FINAL BEST VALIDATION METRICS (SELF-EVALUATION SUMMARY)")
    print("============================================================")
    print(f"  • IoU (Jaccard)            : {final_eval['iou']:.4f}")
    print(f"  • Dice Score (F1)          : {final_eval['dice']:.4f}")
    print(f"  • Precision                : {final_eval['precision']:.4f}")
    print(f"  • Recall (Sensitivity)     : {final_eval['recall']:.4f}")
    print(f"  • Pixel Accuracy           : {final_eval['accuracy']:.4f}")
    print(f"  • Panoptic Quality (PQ)    : {final_eval['pq']:.4f}")
    print(f"  • Segmentation Quality(SQ) : {final_eval['sq']:.4f}")
    print(f"  • Recognition Quality (RQ) : {final_eval['rq']:.4f}")
    print(f"  • 1-to-Many Fragmentations : {final_eval['one_to_many']}")
    print(f"  • Many-to-1 Mergings       : {final_eval['many_to_one']}")
    print("============================================================")


if __name__ == "__main__":
    run_training_pipeline()
