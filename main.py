"""
RobCrop Agricultural Disease Detection
PyTorch Implementation for SIH Hackathon
"""

import torch
from config import Config

def main():
    """Main entry point for RobCrop project"""
    
    print("🌱 Welcome to RobCrop Agricultural Disease Detection!")
    print("=" * 50)
    
    # Print configuration
    Config.print_config()
    
    # Check CUDA setup
    if torch.cuda.is_available():
        print(f"✅ CUDA is ready! Using {torch.cuda.get_device_name(0)}")
        print(f"🚀 Training will be fast with GPU acceleration!")
    else:
        print("⚠️  CUDA not available. Will use CPU (slower training)")
    
    print("\n📁 Project Structure:")
    print("├── config.py     ✅ Configuration ready")
    print("├── main.py       ✅ Entry point ready") 
    print("├── data/         📂 Add your plant disease dataset here")
    print("└── requirements.txt ✅ Dependencies ready")
    
    print("\n🎯 Next Steps:")
    print("1. Add your dataset to the data/ folder")
    print("2. Run: pip install -r requirements.txt")
    print("3. We'll add model.py when ready to build the ResNet50")
    print("4. We'll add train.py when ready to start training")
    
    print("\n🎊 Ready to build your agricultural AI for the hackathon!")

if __name__ == "__main__":
    main()
