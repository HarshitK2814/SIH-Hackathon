"""
RobCrop Data Splitter - Create proper train/val/test splits
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def create_robcrop_splits(source_dir="data/train", output_dir="data", 
                         train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create proper train/val/test splits from your current training data
    
    Args:
        source_dir: Current data/train directory
        output_dir: Output directory for splits
        train_ratio: Proportion for training (0.7 = 70%)
        val_ratio: Proportion for validation (0.15 = 15%) 
        test_ratio: Proportion for testing (0.15 = 15%)
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Target classes for RobCrop
    target_classes = [
        'Pepper,_bell___Bacterial_spot',
        'Pepper,_bell___healthy',
        'Potato___Early_blight',
        'Potato___Late_blight', 
        'Potato___healthy',
        'Tomato___Bacterial_spot',
        'Tomato___Early_blight',
        'Tomato___Late_blight',
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]
    
    print("🔄 Creating RobCrop Train/Val/Test Splits")
    print("=" * 50)
    print(f"📊 Split ratios: Train={train_ratio:.0%}, Val={val_ratio:.0%}, Test={test_ratio:.0%}")
    
    # Create output directories
    splits = ['train', 'val', 'test']
    for split in splits:
        for class_name in target_classes:
            split_class_dir = output_path / split / class_name
            split_class_dir.mkdir(parents=True, exist_ok=True)
    
    total_moved = 0
    split_counts = defaultdict(lambda: defaultdict(int))
    
    # Process each target class
    for class_name in target_classes:
        class_dir = source_path / class_name
        
        if not class_dir.exists():
            print(f"⚠️  Warning: {class_name} not found in source")
            continue
        
        # Get all images for this class
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = []
        for ext in image_extensions:
            images.extend(list(class_dir.glob(f'*{ext}')))
            images.extend(list(class_dir.glob(f'*{ext.upper()}')))
        
        if len(images) == 0:
            print(f"⚠️  No images found for {class_name}")
            continue
        
        # Shuffle images for random split
        random.shuffle(images)
        
        # Calculate split indices
        n_images = len(images)
        train_end = int(n_images * train_ratio)
        val_end = train_end + int(n_images * val_ratio)
        
        # Split images
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # Copy images to respective splits
        for split, split_images in [('train', train_images), 
                                   ('val', val_images), 
                                   ('test', test_images)]:
            for img_path in split_images:
                dest_path = output_path / split / class_name / img_path.name
                shutil.copy2(img_path, dest_path)
                split_counts[split][class_name] += 1
                total_moved += 1
        
        print(f"✅ {class_name:<40}: {len(train_images):>4}/{len(val_images):>4}/{len(test_images):>4} (T/V/Te)")
    
    # Print summary
    print("\n📊 Split Summary:")
    print("=" * 60)
    
    for split in splits:
        total_split = sum(split_counts[split].values())
        print(f"{split.upper():>5}: {total_split:>6} images")
    
    print(f"{'TOTAL':>5}: {total_moved:>6} images moved")
    
    # Verify ratios
    total_images = sum(sum(split_counts[split].values()) for split in splits)
    if total_images > 0:
        actual_ratios = {split: sum(split_counts[split].values()) / total_images 
                        for split in splits}
        
        print(f"\n🎯 Actual Ratios:")
        for split in splits:
            print(f"{split.capitalize():>12}: {actual_ratios[split]:>5.1%}")
    
    return split_counts

def backup_original_data():
    """Backup your original train folder before splitting"""
    source = Path("data/train")
    backup = Path("data/train_original_backup")
    
    if source.exists() and not backup.exists():
        print("💾 Creating backup of original training data...")
        shutil.copytree(source, backup)
        print(f"✅ Backup created: {backup}")
    else:
        print("ℹ️  Backup already exists or source not found")

def main():
    """Create proper academic splits for RobCrop dataset"""
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    print("🧪 RobCrop Academic Data Split Creation")
    print("=" * 50)
    
    # Check if source exists
    if not Path("data/train").exists():
        print("❌ data/train directory not found!")
        return
    
    # Backup original data
    backup_original_data()
    
    # Create new directory structure
    new_data_dir = Path("data_split")
    
    # Create splits (70/15/15 is common academic split)
    split_counts = create_robcrop_splits(
        source_dir="data/train",
        output_dir=str(new_data_dir),
        train_ratio=0.70,
        val_ratio=0.15, 
        test_ratio=0.15
    )
    
    print(f"\n🎊 Academic splits created successfully!")
    print(f"📁 New data structure: {new_data_dir}")
    print(f"📝 Update your config.py to use: DATA_DIR = Path('data_split')")
    
    # Show expected results
    total = sum(sum(counts.values()) for counts in split_counts.values())
    train_count = sum(split_counts['train'].values())
    val_count = sum(split_counts['val'].values())  
    test_count = sum(split_counts['test'].values())
    
    print(f"\n📊 Expected Training Results:")
    print(f"   Training batches: {train_count // 64} (batch_size=64)")
    print(f"   Validation batches: {val_count // 64}")
    print(f"   Test batches: {test_count // 64}")

if __name__ == "__main__":
    main()
