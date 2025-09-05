import torch
from pathlib import Path

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    DATA_DIR = Path("data_split")
    
    # Optimized for RTX 4070
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 30
    IMG_SIZE = 224
    
    # Model settings
    NUM_CLASSES = 11
    PRETRAINED = True
    
    # GPU-specific optimizations
    NUM_WORKERS = 4  # Parallel data loading
    PIN_MEMORY = True  # Faster GPU transfer
    
    # Plant disease classes
    CLASS_NAMES = [
        'Pepper_bell_Bacterial_spot', 'Pepper_bell_healthy',
        'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'Tomato_healthy'
    ]
    
    @classmethod
    def print_config(cls):
        print("🚀 RobCrop PyTorch Configuration")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"Batch Size: {cls.BATCH_SIZE} (optimized for your GPU)")
        print(f"Classes: {cls.NUM_CLASSES}")
        print("=" * 40)
