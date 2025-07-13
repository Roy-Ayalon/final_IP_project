# 🌾 Wheat Head Detection - Final IP Project

Guy Perets 209207117
Roy Ayalon 211326533

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

### Step 2: Create & Activate Virtual Environment

```bash
# Create a wheat-env virtual enviroment
python -m venv wheat-env

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

# Verify installation
pip list | grep torch
```

---

## 📊 Data Preparation

### Option 1: Interactive Setup (Recommended)

Simply run the interactive setup script:

```bash
python setup_project.py
```

This will:
- Guide you through choosing U-Net or DETR (or both)
- Automatically download the Global Wheat Detection dataset from Kaggle
- Set up all necessary folders and files
- Give you the exact commands to start training

### Option 2: Direct Command Line

For more control, use the generate_data.py script directly:

```bash
# Setup for U-Net training with API credentials
python src/generate_data.py --model unet --kaggle_username YOUR_USERNAME --kaggle_key YOUR_API_KEY

# Setup for DETR training with existing kaggle.json
python src/generate_data.py --model detr

# Custom validation split (e.g., 15% validation)
python src/generate_data.py --model unet --val_ratio 0.15 --kaggle_username YOUR_USERNAME --kaggle_key YOUR_API_KEY
```

### Understanding Your Data Structure

Your data should be organized as follows:
```
data_detr/
├── train.csv              # Annotations with bounding boxes
├── train/                 # Training images
│   ├── b6ab77fd7.jpg     # Individual image files
│   ├── 51f1be19e.jpg
│   └── ...
└── test/                  # Test images (for inference)
    ├── 2fd875eaa.jpg
    └── ...

data_unet/
├── masks/                 
├── train/                 # Training data split
│   ├── images/           # Training images
│   │   ├── img_0001.jpg
│   │   ├── img_0002.jpg
│   │   └── ...
│   └── masks/            # Training masks
│       ├── img_0001.jpg
│       ├── img_0002.jpg
│       └── ...
├── val/                   # Validation data split
│   ├── images/           # Validation images
│   │   ├── img_0003.jpg
│   │   ├── img_0004.jpg
│   │   └── ...
│   └── masks/            # Validation masks
│       ├── img_0003.jpg
│       ├── img_0004.jpg
│       └── ...
└── unannotated/           # Images without 
    ├── img_no_wheat_001.jpg
    ├── img_no_wheat_002.jpg
    └── ...

```

---

## 🚀 Training Models

### DETR (Detection Transformer) Training

Train the DETR model with command line arguments:

```bash
# Basic training with default parameters
python src/train_detr.py \
  --csv_path data_detr/train.csv \
  --images_dir data_detr/train \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --num_workers 4

# Advanced training with custom parameters
python src/train_detr.py \
  --csv_path data_detr/train.csv \
  --images_dir data_detr/train \
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
  --images_dir data_unet/train/images \
  --masks_dir data_unet/train/masks \
  --val_images_dir data_unet/val/images \
  --val_masks_dir data_unet/val/masks \
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
├── data_unet/                    # Your wheat detection dataset
├── data_detr/
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
