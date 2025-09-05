# In backend/model.py

import torch
import torch.nn as nn
import torchvision.models as models

class RobCropResNet50(nn.Module):
    """
    This class defines the exact model architecture that matches the saved
    'best_model.pth' checkpoint.
    """
    
    def __init__(self, num_classes=11, pretrained=True, freeze_backbone=True):
        super(RobCropResNet50, self).__init__()
        
        # Load pre-trained ResNet50
        # Use the older 'pretrained=True' to match how the model was likely saved
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get number of features from backbone's original fc layer
        num_features = self.backbone.fc.in_features
        
        # Replace the original fc layer with a simple Linear layer.
        # This seems to be what was done in the original training script.
        self.backbone.fc = nn.Linear(num_features, num_features)
        
        # Define a separate classifier attribute, as indicated by the error logs
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
        
    def forward(self, x):
        """
        The forward pass must also match the original structure.
        """
        # Pass input through the backbone to get features
        features = self.backbone(x)
        
        # Pass those features through the separate classifier
        output = self.classifier(features)
        
        return output
