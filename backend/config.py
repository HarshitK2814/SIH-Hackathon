"""
RobCrop Backend Configuration
Deployment-specific settings for FastAPI backend
"""

import torch
from pathlib import Path
import os

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths (adjusted for backend deployment)
    DATA_DIR = Path("../data_split")  # Relative to backend folder
    MODEL_PATH = Path("../training_outputs/checkpoints")  # Path to trained models
    
    # Model settings optimized for deployment
    BATCH_SIZE = 64
    IMG_SIZE = 224
    NUM_CLASSES = 11
    PRETRAINED = True
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    API_WORKERS = 1  # Single worker for GPU usage
    
    # Performance settings
    MAX_FILE_SIZE_MB = 10  # Maximum upload file size
    MAX_BATCH_SIZE = 10    # Maximum batch prediction size
    REQUEST_TIMEOUT = 30   # Request timeout in seconds
    
    # GPU optimization for deployment
    NUM_WORKERS = 0        # Set to 0 for deployment (avoid multiprocessing issues)
    PIN_MEMORY = True
    
    # Target agricultural classes
    CLASS_NAMES = [
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
    
    # Environment settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def print_config(cls):
        print("🚀 RobCrop Backend Configuration")
        print("=" * 40)
        print(f"Device: {cls.DEVICE}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"API Host: {cls.API_HOST}:{cls.API_PORT}")
        print(f"Model Classes: {cls.NUM_CLASSES}")
        print(f"Debug Mode: {cls.DEBUG}")
        print("=" * 40)
