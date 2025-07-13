#!/usr/bin/env python3
"""
Quick setup script for wheat detection project.
This script automatically downloads the Kaggle dataset and sets up data for training.
"""

import subprocess
import sys
import os

def main():
    print("🌾 Wheat Detection Project - Data Setup")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/generate_data.py"):
        print("❌ Error: Please run this script from the project root directory.")
        print("The script expects to find 'src/generate_data.py'")
        sys.exit(1)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not hasattr(sys, 'base_prefix'):
        # Python 3.3+ venv
        if sys.prefix == sys.base_prefix:
            print("⚠️ Warning: Virtual environment not detected!")
            print("Please activate your virtual environment first:")
            print("  source wheat-env/bin/activate")
            print()
            continue_anyway = input("Continue anyway? (y/N): ").strip().lower()
            if continue_anyway != 'y':
                sys.exit(1)
    
    print("\nChoose which model you want to prepare data for:")
    print("1. U-Net (segmentation-based detection)")
    print("2. DETR (transformer-based object detection)")
    print("3. Both models")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ["1", "2", "3"]:
            break
        print("Please enter 1, 2, or 3")
    
    models = []
    if choice == "1":
        models = ["unet"]
    elif choice == "2":
        models = ["detr"]
    else:
        models = ["unet", "detr"]
    
    # Ask for Kaggle credentials
    print("\n🔑 Kaggle API Credentials")
    print("You can provide credentials in one of these ways:")
    print("1. Enter them now")
    print("2. Use existing ~/.kaggle/kaggle.json file")
    print("3. Use environment variables (KAGGLE_USERNAME, KAGGLE_KEY)")
    
    cred_choice = input("\nEnter your choice (1/2/3): ").strip()
    
    kaggle_username = None
    kaggle_key = None
    
    if cred_choice == "1":
        print("\nGet your API credentials from: https://www.kaggle.com/account")
        kaggle_username = input("Kaggle Username: ").strip()
        kaggle_key = input("Kaggle API Key: ").strip()
        
        if not kaggle_username or not kaggle_key:
            print("❌ Both username and API key are required!")
            sys.exit(1)
    elif cred_choice == "2":
        kaggle_file = os.path.expanduser("~/.kaggle/kaggle.json")
        if not os.path.exists(kaggle_file):
            print("❌ ~/.kaggle/kaggle.json not found!")
            print("Please create it first or choose option 1.")
            sys.exit(1)
        print("✅ Will use existing kaggle.json file")
    elif cred_choice == "3":
        if not (os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY')):
            print("❌ KAGGLE_USERNAME and KAGGLE_KEY environment variables not found!")
            sys.exit(1)
        print("✅ Will use environment variables")
    
    # Ask about validation ratio
    val_ratio = input("\nValidation set ratio (default 0.2): ").strip()
    if not val_ratio:
        val_ratio = "0.2"
    
    try:
        float(val_ratio)
    except ValueError:
        print("Invalid ratio, using default 0.2")
        val_ratio = "0.2"
    
    print(f"\n🚀 Setting up data for: {', '.join(models).upper()}")
    print(f"📊 Validation ratio: {val_ratio}")
    print("\nThis will:")
    print("• Download the Global Wheat Detection dataset from Kaggle")
    print("• Extract and organize the data")
    print("• Create train/validation splits")
    if "unet" in models:
        print("• Generate binary masks for U-Net training")
    print()
    
    # Setup data for each model
    for model in models:
        print(f"\n📦 Setting up data for {model.upper()}...")
        
        cmd = [
            sys.executable, "src/generate_data.py",
            "--model", model,
            "--val_ratio", val_ratio
        ]
        
        # Add Kaggle credentials if provided
        if kaggle_username and kaggle_key:
            cmd.extend(["--kaggle_username", kaggle_username, "--kaggle_key", kaggle_key])
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Error setting up data for {model}: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\n⚠️ Setup interrupted by user")
            sys.exit(1)
    
    print("\n" + "="*50)
    print("✅ Setup complete! Your data is ready for training.")
    
    if "unet" in models:
        print("\n🔸 For U-Net training, the command was displayed above.")
    if "detr" in models:
        print("🔸 For DETR training, the command was displayed above.")
    
    print("\n💡 Tip: Make sure to activate your virtual environment before training!")

if __name__ == "__main__":
    main()
