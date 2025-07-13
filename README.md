# 🌾 Wheat Head Detection - Final IP Project

## � Project Overview

This project implements **wheat head detection** using computer vision and deep learning techniques. The goal is to detect and localize individual wheat heads in field images, which is crucial for agricultural analysis, yield estimation, and crop monitoring.

### 🎯 Task Description

**Objective**: Detect wheat heads in field images using bounding box detection.

**Input**: RGB images of wheat fields (1024×1024 pixels)  
**Output**: Bounding boxes around individual wheat heads with confidence scores

**Data Format**: 
- Images: `.jpg` files containing wheat field photographs
- Annotations: CSV file with columns `[image_id, width, height, bbox, source]`
- Bounding boxes: Format `[x, y, width, height]` in pixel coordinates

**Example annotation**:
```csv
image_id,width,height,bbox,source
b6ab77fd7,1024,1024,"[834.0, 222.0, 56.0, 36.0]",usask_1
b6ab77fd7,1024,1024,"[226.0, 548.0, 130.0, 58.0]",usask_1
```

### 🏗️ Models Implemented

1. **DETR (Detection Transformer)**: State-of-the-art object detection using transformers
2. **U-Net**: Semantic segmentation approach for wheat head detection

---

## 🛠️ Setup Instructions (macOS)

### Step 1: Clone and Navigate to Project

```bash
git clone https://github.com/Roy-Ayalon/final_IP_project.git
cd final_IP_project
```

### Step 2: Activate Virtual Environment

```bash
# Activate the existing wheat-env virtual environment
source wheat-env/bin/activate

# Verify activation (you should see (wheat-env) in your prompt)
which python
```

### Step 3: Install Dependencies

```bash
# Upgrade pip and install requirements
python -m pip install --upgrade pip
pip install -r requirements.txt

# Install additional dependencies for training
pip install pytorch-lightning wandb torchmetrics
pip install segmentation-models-pytorch

# Verify installation
pip list | grep torch
```

---

## 📊 Data Preparation

### Understanding Your Data Structure

Your data should be organized as follows:
```
data/
├── train.csv              # Annotations with bounding boxes
├── train/                 # Training images
│   ├── b6ab77fd7.jpg     # Individual image files
│   ├── 51f1be19e.jpg
│   └── ...
└── test/                  # Test images (for inference)
    ├── 2fd875eaa.jpg
    └── ...
```

### Generate Preprocessed Dataset

Use the data generation script to create masks and split your dataset:

```bash
# Generate masks from bounding boxes and split into train/val
python src/generate_data.py \
  --csv data/train.csv \
  --images_dir data/train \
  --output_masks_dir data/train_masks \
  --train_images_dir data/processed/train \
  --train_masks_dir data/processed/train_masks \
  --val_images_dir data/processed/val \
  --val_masks_dir data/processed/val_masks \
  --unannotated_dir data/unannotated \
  --val_ratio 0.2 \
  --seed 42 \
  --prefix img
```

This will:
- ✅ Generate binary masks from bounding box annotations
- ✅ Split dataset into training (80%) and validation (20%) sets
- ✅ Move images without annotations to `unannotated/` folder
- ✅ Rename files with consistent naming (`img_0001.jpg`, etc.)

---

## 🚀 Training Models

### DETR (Detection Transformer) Training

Train the DETR model with command line arguments:

```bash
# Basic training with default parameters
python src/train_detr.py \
  --csv_path data/train.csv \
  --images_dir data/train \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --num_workers 4

# Advanced training with custom parameters
python src/train_detr.py \
  --csv_path data/train.csv \
  --images_dir data/train \
  --epochs 100 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --num_workers 8 \
  --val_ratio 0.1 \
  --num_queries 100 \
  --hidden_dim 256 \
  --warmup_epochs 5 \
  --project_name "wheat-detection-detr"
```

### U-Net Training (Alternative Approach)

```bash
# Train U-Net for segmentation-based detection
python src/train_unet.py \
  --images_dir data/processed/train \
  --masks_dir data/processed/train_masks \
  --val_images_dir data/processed/val \
  --val_masks_dir data/processed/val_masks \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-4
```

### Training Arguments Reference

| Argument | Description | Default |
|----------|-------------|---------|
| `--csv_path` | Path to CSV annotations file | `data/train.csv` |
| `--images_dir` | Directory containing training images | `data/train` |
| `--epochs` | Number of training epochs | `50` |
| `--batch_size` | Batch size for training | `16` |
| `--learning_rate` | Learning rate for optimizer | `1e-4` |
| `--num_workers` | Number of data loading workers | `4` |
| `--val_ratio` | Validation split ratio | `0.1` |
| `--num_queries` | Number of object queries (DETR) | `100` |
| `--hidden_dim` | Hidden dimension size | `256` |
| `--project_name` | W&B project name | `wheat-detection` |

---

## 📈 Monitoring Training

The training uses **Weights & Biases (W&B)** for experiment tracking:

1. **Login to W&B** (first time only):
   ```bash
   wandb login
   ```

2. **View training progress**:
   - Training loss and validation mAP
   - Sample predictions with bounding boxes
   - Model performance metrics

3. **Access your experiments**: Visit [wandb.ai](https://wandb.ai) to view detailed logs

---

## 🎯 Model Evaluation

Evaluation metrics used:
- **mAP (mean Average Precision)**: Primary metric for object detection
- **IoU thresholds**: 0.50 to 0.75 (Kaggle competition standard)
- **Loss components**: Classification loss + Bounding box regression + GIoU loss

---

## 📁 Project Structure

```
final_IP_project/
├── src/
│   ├── train_detr.py         # DETR training script with CLI
│   ├── train_unet.py         # U-Net training script  
│   ├── Detr.py              # DETR model implementation
│   ├── unet.py              # U-Net model implementation
│   ├── dataset.py           # Dataset classes and data loading
│   ├── loss.py              # Loss functions (Hungarian matching)
│   ├── generate_data.py     # Data preprocessing utilities
│   └── definitions.py       # Configuration constants
├── data/                    # Your wheat detection dataset
├── wheat-env/              # Python virtual environment
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

---

## 🔧 Troubleshooting

### Common Issues on macOS:

1. **Virtual environment not found**:
   ```bash
   # Recreate the environment if needed
   python -m venv wheat-env
   source wheat-env/bin/activate
   ```

2. **CUDA not available**:
   - The code will automatically use MPS (Apple Silicon) or CPU
   - No action needed for M1/M2 Macs

3. **Memory issues**:
   - Reduce `--batch_size` to 8 or 4
   - Reduce `--num_workers` to 2 or 0

4. **Permission errors**:
   ```bash
   chmod +x src/train_detr.py
   ```

---

## 📚 References

- **DETR Paper**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- **Dataset**: [Global Wheat Detection Challenge](https://www.kaggle.com/c/global-wheat-detection)
- **PyTorch Lightning**: [Documentation](https://lightning.ai/docs/pytorch/stable/)
