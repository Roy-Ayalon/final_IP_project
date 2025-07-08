# final_IP_project

## 🛠️ Setup Instructions (using venv)

```bash
# 1. Clone the repository and move into it
git clone https://github.com/Roy-Ayalon/final_IP_project.git
cd final_IP_project

# 2. Create a virtual environment
python -m venv wheat_venv

# 3. Activate the environment
# macOS / Linux:
source wheat_venv/bin/activate
# Windows (cmd or PowerShell):
.\wheat_venv\Scripts\activate

# 4. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 5. Check installed packages
pip list




## 🧪 Preprocessing the Dataset (Mask Generation + Train/Val Split)

This project includes a preprocessing script `src/utils.py` that lets you:
- ✅ Generate binary masks from a CSV of bounding boxes
- ✅ Split the dataset into training and validation sets
- ✅ Rename all images/masks consistently
- 🚫 Move unannotated images to a separate `un_annotated` folder

### 📁 Required Inputs

Make sure you have:
- A folder of images (e.g. `data/train/images`)
- A CSV file with bounding boxes (optional, if you want to generate masks)
- Or existing mask files in `.png` format (e.g. `data/train_masks/*.png`)

---

### ⚙️ Run the preprocessing

#### Option 1 – With CSV (generates masks)

```bash
python src/utils.py \
  --csv data/train.csv \
  --output_masks_dir data/train_masks \
  --images_dir data/train/images \
  --masks_dir data/train_masks \
  --train_images_dir data/real/train/images \
  --train_masks_dir data/real/train/masks \
  --val_images_dir data/real/val/images \
  --val_masks_dir data/real/val/masks \
  --unannotated_dir data/un_annotated \
  --val_ratio 0.2 \
  --prefix sample \
  --seed 42
