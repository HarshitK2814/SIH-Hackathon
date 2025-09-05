"""
RobCrop ResNet50 Model
Optimized for RTX 4070 GPU and Agricultural Disease Detection
"""

import torch
import torch.nn as nn
import torchvision.models as models
from config import Config

class RobCropResNet50(nn.Module):
    """
    ResNet50 model optimized for plant disease detection
    Designed for 11-class agricultural disease classification
    """
    
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True, freeze_backbone=True):
        super(RobCropResNet50, self).__init__()
        
        print("🏗️  Building RobCrop ResNet50...")
        
        # Load pre-trained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone for transfer learning (speeds up training)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("❄️  Backbone frozen for transfer learning")
        
        # Get number of features from backbone
        num_features = self.backbone.fc.in_features  # 2048 for ResNet50
        
        # Remove original classifier by replacing with a Linear layer (keeps type compatibility)
        self.backbone.fc = nn.Linear(num_features, num_features)
        
        # Custom agricultural disease classifier
        self.classifier = nn.Sequential(
            # First layer with heavy dropout for regularization
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            
            # Second layer
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            
            # Third layer
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Final classification layer
            nn.Linear(128, num_classes)
        )
        
        # Initialize custom layers
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize classifier weights using best practices"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Extract features using ResNet50 backbone
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def unfreeze_backbone(self, layers_to_unfreeze=2):
        """
        Unfreeze last few layers of backbone for fine-tuning
        layers_to_unfreeze: number of layer blocks to unfreeze (1-4)
        """
        layers = [self.backbone.layer4, self.backbone.layer3, 
                 self.backbone.layer2, self.backbone.layer1]
        
        for i in range(min(layers_to_unfreeze, len(layers))):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        print(f"🔓 Unfroze last {layers_to_unfreeze} layer blocks for fine-tuning")
    
    def get_model_info(self):
        """Get detailed model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': frozen_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        }

def create_model(device=None):
    """
    Create and initialize the RobCrop model
    Returns model ready for training on specified device
    """
    if device is None:
        device = Config.DEVICE
    
    print("🎯 Creating RobCrop ResNet50 for Agricultural Disease Detection")
    
    # Create model
    model = RobCropResNet50(
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        freeze_backbone=True
    )
    
    # Move to device (GPU/CPU)
    model = model.to(device)
    
    # Get model information
    info = model.get_model_info()
    
    print("📊 Model Statistics:")
    print(f"   Total parameters: {info['total_params']:,}")
    print(f"   Trainable parameters: {info['trainable_params']:,}")
    print(f"   Frozen parameters: {info['frozen_params']:,}")
    print(f"   Model size: {info['model_size_mb']:.1f} MB")
    print(f"   Device: {device}")
    
    # Test model with dummy input
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
        output = model(dummy_input)
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print("✅ Model created successfully!")
    
    return model

def test_model_gpu_memory():
    """Test model memory usage on GPU"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available for memory testing")
        return
    
    print("🧪 Testing GPU memory usage...")
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Record initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    # Create model
    model = create_model()
    
    # Record memory after model creation
    model_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    # Test with batch
    model.train()
    dummy_batch = torch.randn(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    
    # Forward pass
    with torch.cuda.device(0):
        output = model(dummy_batch)
        batch_memory = torch.cuda.memory_allocated() / 1024**3  # GB
    
    print("💾 GPU Memory Usage:")
    print(f"   Initial: {initial_memory:.2f} GB")
    print(f"   Model: {model_memory - initial_memory:.2f} GB")
    print(f"   Batch processing: {batch_memory - model_memory:.2f} GB")
    print(f"   Total used: {batch_memory:.2f} GB")
    
    # Get GPU info
    gpu_properties = torch.cuda.get_device_properties(0)
    total_memory = gpu_properties.total_memory / 1024**3
    print(f"   GPU total memory: {total_memory:.1f} GB")
    print(f"   Memory utilization: {(batch_memory/total_memory)*100:.1f}%")
    
    if batch_memory / total_memory < 0.8:
        print("✅ Memory usage looks good for training!")
    else:
        print("⚠️  High memory usage - consider reducing batch size")
    
    # Cleanup
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
    
    # Test model creation
    model = create_model()
    
    # Test GPU memory if CUDA available
    if torch.cuda.is_available():
        test_model_gpu_memory()
    
    # Print model summary
    ModelSummary.print_layer_details(model)
    
    print("\n🎊 Model ready for training!")
    print("📝 Next step: Create data_loader.py for dataset handling")

if __name__ == "__main__":
    main()
