# 🗄️ Dataset Setup Guide — MAGFiLO 1.0 (Solar Filament Segmentation 2026)

This directory contains documentation and metadata for the **MAGFiLO 1.0 (Manually Annotated GONG Filaments in H-$\alpha$ Observations)** dataset used in the **Solar Filament Segmentation Challenge 2026**.

Because the raw high-resolution ($2048 \times 2048$) telescope imagery is several gigabytes in size, the raw image folders (`train_images/` and `test_images/`) are excluded from Git tracking via `.gitignore`. However, the exact **Train/Validation split manifest** is tracked in [`../pipeline_results/splits/split_manifest.csv`](../pipeline_results/splits/split_manifest.csv) to guarantee $100\%$ reproducibility.

---

## 📥 1. How to Download the Dataset

### Option A: Using the Official Kaggle CLI
1. Install the Kaggle CLI and configure your `kaggle.json` API token:
   ```bash
   pip install kaggle
   ```
2. Download and unzip the competition dataset:
   ```bash
   kaggle competitions download -c filament-segmentation-2026 -p ./MAGFiLO_1.0_Kaggle_2026
   unzip ./MAGFiLO_1.0_Kaggle_2026/filament-segmentation-2026.zip -d ./MAGFiLO_1.0_Kaggle_2026
   ```

### Option B: Manual Download
1. Visit the **Solar Filament Segmentation 2026** competition page on Kaggle (or the official GONG / MAGFiLO data portal).
2. Download the dataset archive and extract it on your local machine or Google Drive.

---

## 📂 2. Expected Directory Structure

After extracting the dataset, ensure your dataset folder is structured as follows:

```
MAGFiLO_1.0_Kaggle_2026/
├── MAGFiLO_1.0_Annotations_kaggle2026_train.json   # Official COCO-format instance annotations
├── train_images/                                   # H-alpha training & validation images (.jpg / .png)
│   ├── 20120605141234Ch.jpg
│   ├── ...
└── test_images/                                    # Unlabeled H-alpha test images for RLE submission
    ├── ...
```

---

## 🔧 3. Configuring Paths in the Code

### In `train_and_evaluate_colab.py`:
Update the `Config` class (Cell 3) to point to your extracted dataset location:
```python
class Config:
    IMAGE_DIR = r"C:\Users\hp\Downloads\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026\train_images"
    JSON_PATH = r"C:\Users\hp\Downloads\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026\MAGFiLO_1.0_Annotations_kaggle2026_train.json"
```

### In `inference_test_colab.py`:
Update the `InferenceConfig` class (Cell 2) to point to your test images folder and trained checkpoint:
```python
class InferenceConfig:
    TEST_IMAGE_DIR = r"C:\Users\hp\Downloads\filament-segmentation-2026\MAGFiLO_1.0_Kaggle_2026\test_images"
    CHECKPOINT_PATH = r".\pipeline_results\best_attention_unet.pth"
```

---

## 🛡️ 4. Event-Grouped Leakage-Free Split Methodology

Solar H-$\alpha$ telescopes capture multiple observations on the same solar day (`YYYYMMDD`). A naive random split across filenames allows temporally adjacent captures of the exact same filament structures to appear in both the training and validation sets, inflating validation metrics via temporal data leakage.

To prevent this, `create_leakage_free_split()` in `train_and_evaluate_colab.py`:
1. Filters out any pre-augmented image copies (`_aug`, `_rot`, `_flip`).
2. Extracts the 8-digit observation date key (`YYYYMMDD`) from every physical filename using regular expressions.
3. Groups all captures from the same solar day together and assigns entire solar days exclusively to either **Train** or **Validation** ($110$ validation images across zero overlapping dates).
4. Exports the verified split to [`../pipeline_results/splits/split_manifest.csv`](../pipeline_results/splits/split_manifest.csv).