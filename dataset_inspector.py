"""
Dataset Inspector - Check your actual dataset structure
"""
from pathlib import Path
import os

def inspect_dataset():
    """Inspect actual dataset structure"""
    data_dir = Path("data")
    
    print("🔍 Inspecting Your Actual Dataset Structure")
    print("=" * 50)
    
    if not data_dir.exists():
        print("❌ data/ directory not found!")
        return
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        print(f"\n📂 {split.upper()} Directory:")
        
        if not split_dir.exists():
            print(f"   ❌ {split_dir} not found")
            continue
        
        # List all subdirectories (classes)
        subdirs = [d for d in split_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            print(f"   ⚠️  No class directories found in {split_dir}")
            continue
        
        print(f"   Found {len(subdirs)} classes:")
        for i, class_dir in enumerate(sorted(subdirs)):
            # Count images in this class
            image_count = 0
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                image_count += len(list(class_dir.glob(f'*{ext}')))
                image_count += len(list(class_dir.glob(f'*{ext.upper()}')))
            
            print(f"   {i:2d}. {class_dir.name:<40}: {image_count:>6} images")

if __name__ == "__main__":
    inspect_dataset()
