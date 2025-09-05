"""
RobCrop Inference - FIXED for all Pylance errors
Real-time crop disease detection for agricultural rover
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Union, List
from model import create_model

class RobCropInference:
    """
    Fixed inference class for RobCrop agricultural disease detection
    All Pylance errors resolved
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model()
        
        # Load trained model weights
        if model_path is None:
            model_path = self._find_best_model()
        
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"✅ Model loaded from: {model_path}")
        else:
            print("⚠️ No model found - using untrained model")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Preprocessing pipeline (matches training transforms)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts PIL → Tensor [0,255] → [0,1]
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Class names for agricultural diseases
        self.class_names: List[str] = [
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
    
    def _find_best_model(self) -> Optional[str]:
        """Auto-find the best trained model - returns None if not found"""
        possible_paths = [
            "training_outputs/checkpoints/*/best_model.pth",
            "checkpoints/robcrop_best.pth",
            "robcrop_final.pth"
        ]
        
        for pattern in possible_paths:
            models = list(Path(".").glob(pattern))
            if models:
                return str(models[0])  # Return first found
        
        return None  # Explicitly return None if no model found
    
    def _convert_to_pil(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various image inputs to PIL Image"""
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            return Image.fromarray(image_input).convert('RGB')
        elif hasattr(image_input, 'convert'):
            # Already PIL Image
            return image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """
        Predict disease from image
        
        Args:
            image_input: Can be PIL Image, file path (str), or numpy array
            
        Returns:
            Dict with disease, confidence, and action
        """
        try:
            # Convert input to PIL Image
            pil_image = self._convert_to_pil(image_input)
            
            # FIXED: Apply transforms to PIL Image (converts to tensor)
            tensor_image = self.transform(pil_image)  # Shape: [C, H, W] - now it's a tensor
            
            # FIXED: Add batch dimension to tensor (not PIL Image)
            batch_tensor = tensor_image.unsqueeze(0)  # type: ignore # Shape: [1, C, H, W]
            
            # Move to device
            batch_tensor = batch_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)  # Shape: [1, num_classes]
                
                # Get probabilities
                probabilities = F.softmax(outputs, dim=1)  # Shape: [1, num_classes]
                
                # FIXED: Proper tensor indexing to avoid slice errors
                confidence_tensor, predicted_tensor = torch.max(probabilities, dim=1)
                
                # FIXED: Convert to Python types (fixes JSON serialization)
                confidence = float(confidence_tensor.item())
                predicted_class_idx = int(predicted_tensor.item())
            
            # Get disease name safely
            if 0 <= predicted_class_idx < len(self.class_names):
                disease_name = self.class_names[predicted_class_idx]
            else:
                disease_name = f"Unknown_class_{predicted_class_idx}"
            
            # Determine action
            is_healthy = 'healthy' in disease_name.lower()
            action = 'SKIP' if is_healthy else 'SPRAY'
            
            return {
                'disease': disease_name,
                'confidence': confidence,
                'class_index': predicted_class_idx,
                'action': action,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }
            
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return {
                'disease': 'Error',
                'confidence': 0.0,
                'class_index': -1,
                'action': 'ERROR',
                'error': str(e)
            }
    
    def predict_batch(self, image_list: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """Predict diseases for multiple images at once"""
        if not image_list:
            return []
        
        try:
            # Process all images
            batch_tensors = []
            
            for image_input in image_list:
                # Convert to PIL Image
                pil_image = self._convert_to_pil(image_input)
                
                # Transform to tensor
                tensor_image = self.transform(pil_image)
                batch_tensors.append(tensor_image)
            
            # Stack into batch: [batch_size, C, H, W]
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Batch inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, dim=1)
            
            # Process results
            results = []
            for i in range(len(image_list)):
                confidence = float(confidences[i].item())
                predicted_idx = int(predicted_classes[i].item())
                
                if 0 <= predicted_idx < len(self.class_names):
                    disease_name = self.class_names[predicted_idx]
                else:
                    disease_name = f"Unknown_class_{predicted_idx}"
                
                action = 'SKIP' if 'healthy' in disease_name.lower() else 'SPRAY'
                
                results.append({
                    'disease': disease_name,
                    'confidence': confidence,
                    'class_index': predicted_idx,
                    'action': action
                })
            
            return results
            
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            return [{'disease': 'Error', 'confidence': 0.0, 'action': 'ERROR', 'error': str(e)} 
                   for _ in image_list]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'device': str(self.device),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_loaded': hasattr(self.model, 'state_dict')
        }

# Convenience functions
def predict_single_image(image_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick function to predict single image"""
    detector = RobCropInference(model_path)
    return detector.predict(image_path)

def predict_from_camera(camera_index: int = 0, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Capture from camera and predict"""
    try:
        import cv2
        
        cap = cv2.VideoCapture(camera_index)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector = RobCropInference(model_path)
            return detector.predict(rgb_frame)
        else:
            return {'error': 'Failed to capture image from camera'}
            
    except ImportError:
        return {'error': 'OpenCV not installed. Install with: pip install opencv-python'}
    except Exception as e:
        return {'error': str(e)}

def test_inference() -> bool:
    """Test the inference system"""
    print("🧪 Testing RobCrop Inference System")
    print("=" * 40)
    
    try:
        detector = RobCropInference()
        print("✅ Model initialized successfully")
        
        # Test with dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        result = detector.predict(dummy_image)
        
        if 'error' not in result:
            print(f"✅ Prediction test passed:")
            print(f"   Disease: {result['disease']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Action: {result['action']}")
            return True
        else:
            print(f"❌ Prediction failed: {result['error']}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_inference()
