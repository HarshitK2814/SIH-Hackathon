"""
RobCrop Data Loader - FIXED VERSION
Optimized for 8GB VRAM with correct transform ordering
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import joblib
from collections import Counter
import os
from config import Config

# Your target 11 agricultural classes (exact names from your dataset)
TARGET_CLASSES = [
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

class RobCropFilteredDataset(Dataset):
    """
    Filtered dataset for RobCrop 11-class agricultural disease detection
    Optimized for 8GB VRAM with joblib caching
    """
    
    def __init__(self, data_dir, transform=None, use_cache=True, cache_dir="cache"):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.target_classes = TARGET_CLASSES
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create class mapping: class_name → index (0-10)
        self.class_mapping = {name: idx for idx, name in enumerate(self.target_classes)}
        
        # Cache file path
        cache_file = self.cache_dir / f"{self.data_dir.name}_robcrop_filtered_cache.pkl"
        
        print(f"📂 Loading RobCrop dataset from: {self.data_dir}")
        print(f"🎯 Filtering for {len(self.target_classes)} agricultural classes...")
        
        if self.use_cache and cache_file.exists():
            print("⚡ Loading cached filtered dataset...")
            try:
                cached_data = joblib.load(cache_file)
                self.image_paths = cached_data['image_paths']
                self.labels = cached_data['labels']
                self.class_counts = cached_data['class_counts']
                print("✅ Filtered cache loaded successfully!")
            except Exception as e:
                print(f"⚠️ Cache corrupted, rebuilding: {e}")
                self.image_paths, self.labels, self.class_counts = self._load_filtered_dataset()
        else:
            print("🔍 Scanning and filtering dataset...")
            self.image_paths, self.labels, self.class_counts = self._load_filtered_dataset()
            
            if self.use_cache and len(self.image_paths) > 0:
                cache_data = {
                    'image_paths': self.image_paths,
                    'labels': self.labels,
                    'class_counts': self.class_counts
                }
                joblib.dump(cache_data, cache_file, compress=3)
                print(f"💾 Filtered dataset cached to: {cache_file}")
        
        print(f"📊 RobCrop dataset loaded: {len(self.image_paths)} images across {len(self.class_counts)} classes")
        if len(self.image_paths) > 0:
            self._print_class_distribution()
    
    def _load_filtered_dataset(self):
        """Load and filter dataset for only target agricultural classes"""
        image_paths = []
        labels = []
        class_counts = Counter()
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Get all subdirectories in data folder
        all_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        print("\n🔍 Scanning classes:")
        for class_dir in sorted(all_dirs):
            class_name = class_dir.name
            
            # Check if this class is in our target list
            if class_name in self.class_mapping:
                target_idx = self.class_mapping[class_name]
                
                # Find all valid image files
                class_images = []
                for ext in valid_extensions:
                    class_images.extend(list(class_dir.glob(f'*{ext}')))
                    class_images.extend(list(class_dir.glob(f'*{ext.upper()}')))
                
                # Add to dataset
                for img_path in class_images:
                    image_paths.append(str(img_path))
                    labels.append(target_idx)
                    class_counts[class_name] += 1
                
                print(f"   ✅ {target_idx:2d}. {class_name:<40}: {len(class_images):>5} images")
        
        return image_paths, labels, class_counts
    
    def _print_class_distribution(self):
        """Print class distribution for target classes"""
        print("\n📋 RobCrop Class Distribution:")
        print("=" * 70)
        
        total_images = len(self.image_paths)
        for i, class_name in enumerate(self.target_classes):
            count = self.class_counts.get(class_name, 0)
            percentage = (count / total_images * 100) if total_images > 0 else 0
            print(f"   {i:2d}. {class_name:<40}: {count:>5} ({percentage:>5.1f}%)")
        
        print("=" * 70)
        print(f"   Total RobCrop images: {total_images}")
        
        # Check for class imbalance
        counts = list(self.class_counts.values())
        if len(counts) > 1:
            imbalance_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
            if imbalance_ratio > 3:
                print(f"⚠️  Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)")
                print("💡 Consider using WeightedRandomSampler")
            else:
                print("✅ Classes are reasonably balanced")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if len(self.image_paths) == 0:
            # Return dummy data for empty dataset
            dummy_image = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)
            return dummy_image, 0
        
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced dataset"""
        if len(self.image_paths) == 0:
            return torch.ones(len(self.target_classes))
        
        # Count samples per class
        class_sample_counts = [0] * len(self.target_classes)
        for label in self.labels:
            class_sample_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.image_paths)
        num_classes = len(self.target_classes)
        
        class_weights = []
        for count in class_sample_counts:
            if count > 0:
                weight = total_samples / (num_classes * count)
            else:
                weight = 0.0
            class_weights.append(weight)
        
        return torch.tensor(class_weights, dtype=torch.float32)

def get_robcrop_transforms():
    """
    FIXED: Get optimized transforms with correct ordering
    PIL transforms first, then ToTensor, then tensor transforms
    """
    
    # Training transforms - FIXED ORDER
    train_transform = transforms.Compose([
        # 1. PIL Image transforms (applied to PIL Images)
        transforms.Resize((256, 256)),  
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        
        # Color augmentation (important for plant diseases)
        transforms.ColorJitter(
            brightness=0.3,    # Lighting conditions
            contrast=0.3,      # Different contrast levels
            saturation=0.2,    # Leaf color variations
            hue=0.1           # Slight hue shifts
        ),
        
        # Geometric augmentation
        transforms.RandomAffine(
            degrees=15,           # Rotation
            translate=(0.1, 0.1), # Translation
            scale=(0.9, 1.1),     # Scaling
            shear=10              # Shearing
        ),
        
        # 2. Convert PIL Image to Tensor
        transforms.ToTensor(),
        
        # 3. Tensor transforms (applied to tensors)
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33)),  # NOW CORRECTLY PLACED!
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet means
            std=[0.229, 0.224, 0.225]    # ImageNet stds
        )
    ])
    
    # Validation/Test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform

def create_robcrop_data_loaders(use_cache=True, use_weighted_sampler=False):
    """
    Create optimized data loaders for RobCrop training on 8GB VRAM
    Returns only the 11 target agricultural classes
    """
    
    print("🚀 Creating RobCrop Data Loaders (8GB VRAM Optimized)")
    print(f"🎯 Target classes: {len(TARGET_CLASSES)} agricultural diseases")
    
    # Get FIXED transforms
    train_transform, val_transform = get_robcrop_transforms()
    
    # Create datasets for each split
    datasets = {}
    loaders = {}
    
    splits_transforms = {
        'train': train_transform,
        'val': val_transform, 
        'test': val_transform
    }
    
    for split, transform in splits_transforms.items():
        data_path = Config.DATA_DIR / split
        
        if data_path.exists():
            print(f"\n📂 Processing {split} split...")
            dataset = RobCropFilteredDataset(
                data_path, 
                transform=transform,
                use_cache=use_cache
            )
            
            if len(dataset) > 0:
                datasets[split] = dataset
                
                # Configure sampler for training
                sampler = None
                shuffle = True
                
                if split == 'train' and use_weighted_sampler:
                    sampler = create_weighted_sampler(dataset)
                    shuffle = False  # Don't shuffle when using sampler
                
                # Optimized for 8GB VRAM
                batch_size = 64 if split == 'train' else 64  # Conservative for 8GB
                num_workers = 4 if len(dataset) > 1000 else 2  # Adjust workers based on dataset size
                
                # Create data loader
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=True,           # Faster CPU→GPU transfer
                    persistent_workers=True,  # Keep workers alive between epochs
                    prefetch_factor=2,        # Pre-load batches
                    drop_last=(split == 'train')  # Drop last incomplete batch for training
                )
                
                loaders[split] = loader
                print(f"✅ {split.capitalize()} loader: {len(dataset)} samples, {len(loader)} batches (batch_size={batch_size})")
            else:
                print(f"⚠️ {split.capitalize()} dataset is empty after filtering")
        else:
            print(f"⚠️ {data_path} not found, skipping {split} split")
    
    return loaders, datasets

def create_weighted_sampler(dataset):
    """Create weighted sampler for balanced training on imbalanced data"""
    if len(dataset) == 0:
        return None
    
    class_weights = dataset.get_class_weights()
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("⚖️  Created weighted sampler for balanced training")
    return sampler

def analyze_robcrop_dataset():
    """Analyze the filtered RobCrop dataset"""
    print("📊 RobCrop Dataset Analysis (11 Agricultural Classes)")
    print("=" * 60)
    
    try:
        # Create loaders
        loaders, datasets = create_robcrop_data_loaders(use_cache=True)
        
        if not loaders:
            print("❌ No datasets found after filtering!")
            return
        
        total_images = 0
        
        for split, dataset in datasets.items():
            num_images = len(dataset)
            total_images += num_images
            
            loader = loaders[split]
            num_batches = len(loader)
            
            print(f"\n{split.upper()} SET:")
            print(f"   Agricultural images: {num_images:,}")
            print(f"   Batches: {num_batches:,}")
            print(f"   Batch size: 64")
        
        print(f"\n🎯 ROBCROP SUMMARY:")
        print(f"   Total agricultural images: {total_images:,}")
        print(f"   Classes: {len(TARGET_CLASSES)} (filtered from 38)")
        print(f"   Optimization: 8GB VRAM (RTX 4070)")
        
        # Test loading speed
        if 'train' in loaders:
            print(f"\n⏱️  Testing loading speed...")
            import time
            
            train_loader = loaders['train']
            start_time = time.time()
            
            # Load first 3 batches to test
            for i, (images, labels) in enumerate(train_loader):
                if i >= 2:  # Test 3 batches
                    break
            
            elapsed = time.time() - start_time
            batches_per_sec = 3 / elapsed
            images_per_sec = batches_per_sec * 64
            
            print(f"   Speed: {images_per_sec:.0f} images/second")
            print(f"   Batch loading: {batches_per_sec:.1f} batches/second")
            print("✅ Transform ordering fixed - no more errors!")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def clear_robcrop_cache():
    """Clear all RobCrop cached data"""
    cache_dir = Path("cache")
    if cache_dir.exists():
        import shutil
        cache_files = list(cache_dir.glob("*robcrop*cache.pkl"))
        
        if cache_files:
            for cache_file in cache_files:
                cache_file.unlink()
            print(f"🗑️  Cleared {len(cache_files)} RobCrop cache files")
        else:
            print("ℹ️  No RobCrop cache files to clear")
    else:
        print("ℹ️  No cache directory found")

def main():
    """Test the FIXED RobCrop data loader"""
    print("🧪 Testing FIXED RobCrop Agricultural Data Loader")
    print("🎯 Optimized for 8GB VRAM RTX 4070")
    print("=" * 50)
    
    try:
        # Analyze dataset
        analyze_robcrop_dataset()
        
        print(f"\n🎊 RobCrop data loader FIXED and ready!")
        print(f"✅ Transform ordering corrected")
        print(f"✅ 11 agricultural classes filtered and optimized") 
        print(f"⚡ 8GB VRAM friendly (batch_size=64)")
        print(f"💾 Joblib caching for fast subsequent loads")
        print(f"📝 Ready for train.py implementation")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
