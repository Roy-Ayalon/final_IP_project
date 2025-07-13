# 🌾 Quick Start Guide

## Automatic Data Setup

This project now automatically downloads and sets up the data for you! No manual downloads needed.

### Prerequisites

1. **Activate Virtual Environment**:
   ```bash
   # Navigate to project directory
   cd final_IP_project
   
   # Activate the virtual environment
   source wheat-env/bin/activate
   ```

2. **Kaggle API Setup** (Choose one option):

   **Option A: Provide credentials via command line**
   ```bash
   # Get your credentials from https://www.kaggle.com/account
   python src/generate_data.py --model unet --kaggle_username YOUR_USERNAME --kaggle_key YOUR_API_KEY
   ```

   **Option B: Use kaggle.json file**
   ```bash
   # 1. Go to https://www.kaggle.com/account
   # 2. Click "Create New API Token"
   # 3. Move downloaded kaggle.json to ~/.kaggle/kaggle.json
   # 4. Set permissions: chmod 600 ~/.kaggle/kaggle.json
   python src/generate_data.py --model unet
   ```

   **Option C: Use environment variables**
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   python src/generate_data.py --model unet
   ```

3. **Accept Competition Rules**:
   - Visit https://www.kaggle.com/c/global-wheat-detection/rules
   - Click "I Understand and Accept" to join the competition

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

## What Gets Created

### For U-Net Training:
```
data_unet/
├── train/
│   ├── images/     # Training images
│   └── masks/      # Training masks (binary)
├── val/
│   ├── images/     # Validation images  
│   └── masks/      # Validation masks
├── masks/          # Generated masks from CSV
└── unannotated/    # Images without annotations
```

### For DETR Training:
```
data_detr/
├── train/          # All training images
└── train.csv       # Annotations file
```

## Next Steps

After running the setup, you'll get the exact training commands. For example:

```bash
# U-Net training
python src/train_unet.py --images_dir data_unet/train/images --masks_dir data_unet/train/masks --val_images_dir data_unet/val/images --val_masks_dir data_unet/val/masks

# DETR training  
python src/train_detr.py --csv_path data_detr/train.csv --images_dir data_detr/train
```

## Troubleshooting

- **"Kaggle API credentials not found"**: Follow the Kaggle API setup steps above
- **"Failed to download dataset"**: Make sure you've accepted the competition rules
- **Permission errors**: Ensure kaggle.json has correct permissions (600)

That's it! The script handles everything else automatically. 🎉
