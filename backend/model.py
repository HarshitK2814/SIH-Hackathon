"""
RobCrop ResNet50 Model
Optimized for RTX 4070 GPU and Agricultural Disease Detection
"""
import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os

# --- FIX: Add path to import config.py correctly ---
# This block allows this script to find files like 'config.py' in the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ----------------------------------------------------
from config import Config

class RobCropResNet50(nn.Module):
    """
    ResNet50 model optimized for plant disease detection.
    This architecture EXACTLY matches the saved checkpoint file to prevent loading errors.
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True, freeze_backbone=True):
        super(RobCropResNet50, self).__init__()
        
        print("🏗️  Building RobCrop ResNet50...")
        
        # Using the older 'pretrained=True' parameter to match how the model was likely saved
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone layers
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("❄️  Backbone frozen for transfer learning")
        
        num_features = self.backbone.fc.in_features
        
        # This architecture with a separate 'classifier' attribute matches your checkpoint
        self.backbone.fc = nn.Linear(num_features, num_features)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights using best practices"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the backbone and then the classifier"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def unfreeze_backbone(self, layers_to_unfreeze=2):
        """Unfreeze last few layers of backbone for fine-tuning"""
        layers = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
        for i in range(min(layers_to_unfreeze, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = True
        print(f"🔓 Unfroze last {layers_to_unfreeze} layer blocks for fine-tuning")
    
    def get_model_info(self):
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'model_size_mb': total_params * 4 / (1024**2)
        }

def create_model(device=None):
    """Factory function to create, initialize, and move the model to the correct device."""
    if device is None:
        device = Config.DEVICE
    
    print("🎯 Creating RobCrop ResNet50 for Agricultural Disease Detection")
    
    model = RobCropResNet50(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        freeze_backbone=True
    ).to(device)
    
    # This check is good practice
    if hasattr(model, 'get_model_info'):
        info = model.get_model_info()
        print("📊 Model Statistics:")
        print(f"   Total parameters: {info['total_params']:,}")
        print(f"   Trainable parameters: {info['trainable_params']:,}")
        print(f"   Frozen parameters: {info['frozen_params']:,}")
        print(f"   Model size: {info['model_size_mb']:.2f} MB")
        print(f"   Device: {device}")

    model.eval()
    return model

def test_model_gpu_memory():
    """Test model memory usage on GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available for memory testing")
        return
    
    print("🧪 Testing GPU memory usage...")
    
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    model = create_model()
    model_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    model.train()
    dummy_batch = torch.randn(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    
    with torch.cuda.device(0):
        output = model(dummy_batch)
        batch_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    print("💾 GPU Memory Usage:")
    print(f"   Initial: {initial_memory:.2f} GB")
    print(f"   Model: {model_memory - initial_memory:.2f} GB")
    print(f"   Batch processing: {batch_memory - model_memory:.2f} GB")
    print(f"   Total used: {batch_memory:.2f} GB")
    
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory = gpu_properties.total_memory / 1024**3
    print(f"   GPU total memory: {total_memory:.1f} GB")
    print(f"   Memory utilization: {(batch_memory/total_memory)*100:.1f}%")
    
    if batch_memory / total_memory < 0.8:
        print("✅ Memory usage looks good for training!")
    else:
        print("⚠️  High memory usage - consider reducing batch size")
    
    del model, dummy_batch, output
    torch.cuda.empty_cache()

class ModelSummary:
    """Utility class for model analysis"""
    
    @staticmethod
    def print_layer_details(model):
        """Print detailed layer information"""
        print("\n🔍 Layer Details:")
        print("=" * 60)
        
        total_params = 0
        trainable_params = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                num_params = sum(p.numel() for p in module.parameters())
                num_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                if num_params > 0:
                    status = "🔓" if num_trainable > 0 else "❄️"
                    print(f"{status} {name:<30} {str(module):<40} {num_params:>10,}")
                    
                total_params += num_params
                trainable_params += num_trainable
        
        print("=" * 60)
        print(f"Total: {total_params:,} | Trainable: {trainable_params:,}")
    
    @staticmethod
    def analyze_gradient_flow(model):
        """Check which layers will receive gradients"""
        print("\n🌊 Gradient Flow Analysis:")
        print("=" * 40)
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"✅ {name}")
            else:
                print(f"❌ {name}")

def main():
    """Test the model creation and functionality"""
    print("🧪 Testing RobCrop ResNet50 Model")
    print("=" * 50)
    
    model = create_model()
    
    if torch.cuda.is_available():
        test_model_gpu_memory()
    
    ModelSummary.print_layer_details(model)
    
    print("\n🎊 Model ready for training!")

if __name__ == "__main__":
    main()
