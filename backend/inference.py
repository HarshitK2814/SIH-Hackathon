"""
RobCrop Inference - FINAL CORRECTED VERSION
Real-time crop disease detection for agricultural rover
"""
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Union, List
import cv2 # Import OpenCV for camera functionality

# This will import create_model from the model.py file in the same folder
from model import create_model

class RobCropInference:
    """
    Inference class for RobCrop agricultural disease detection.
    This version is corrected to reliably find and load the model.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = create_model(device=self.device)
        
        # --- FIX: More robust model path finding ---
        if model_path is None:
            # Construct the absolute path to the model relative to this script's location
            current_dir = Path(__file__).parent
            model_path_obj = current_dir / "training_outputs" / "checkpoints" / "robcrop_20250904_231418" / "best_model.pth"
        else:
            model_path_obj = Path(model_path)
        # ---------------------------------------------
        
        if model_path_obj.exists():
            # --- FIX: Correctly load the state dict from the checkpoint ---
            checkpoint = torch.load(str(model_path_obj), map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint) # Fallback for raw state_dict files
            # ----------------------------------------------------------------
            print(f"✅ Model loaded from: {model_path_obj}")
        else:
            print(f"⚠️ Model not found at {model_path_obj} - using untrained model")
        
        self.model.eval()
        self.model.to(self.device)
        
        # This pre-processing pipeline is correct and MUST match training
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # This class name list must be in the exact order the model was trained on
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
    
    def _convert_to_pil(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Convert various image inputs to a standard PIL Image"""
        if isinstance(image_input, str):
            return Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Assumes the numpy array is in RGB format if coming from a non-OpenCV source
            return Image.fromarray(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            return image_input.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
    
    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Predict disease from a single image."""
        try:
            pil_image = self._convert_to_pil(image_input)
            tensor_image = self.transform(pil_image)
            batch_tensor = tensor_image.unsqueeze(0).to(self.device) # type: ignore
            
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence_tensor, predicted_tensor = torch.max(probabilities, dim=1)
                
                confidence = float(confidence_tensor.item())
                predicted_class_idx = int(predicted_tensor.item())
            
            disease_name = self.class_names[predicted_class_idx]
            action = 'SKIP' if 'healthy' in disease_name.lower() else 'SPRAY'
            
            return {
                'disease': disease_name,
                'confidence': confidence,
                'class_index': predicted_class_idx,
                'action': action,
            }
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return {'disease': 'Error', 'confidence': 0.0, 'action': 'ERROR', 'error': str(e)}
    
    def predict_batch(self, image_list: List[Union[str, np.ndarray, Image.Image]]) -> List[Dict[str, Any]]:
        """Predict diseases for a batch of images."""
        if not image_list:
            return []
        
        try:
            batch_tensors = [self.transform(self._convert_to_pil(img)) for img in image_list]
            batch_tensors = [tensor if isinstance(tensor, torch.Tensor) else transforms.ToTensor()(tensor) for tensor in batch_tensors]
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(batch)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_indices = torch.max(probabilities, dim=1)
            
            results = []
            for i in range(len(image_list)):
                idx = int(predicted_indices[i].item())
                disease_name = self.class_names[idx]
                results.append({
                    'disease': disease_name,
                    'confidence': confidences[i].item(),
                    'class_index': idx,
                    'action': 'SKIP' if 'healthy' in disease_name.lower() else 'SPRAY'
                })
            return results
        except Exception as e:
            print(f"❌ Batch prediction error: {str(e)}")
            return [{'disease': 'Error', 'confidence': 0.0, 'action': 'ERROR', 'error': str(e)} for _ in image_list]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Returns diagnostic information about the model."""
        return {
            'device': str(self.device),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'model_loaded': hasattr(self.model, 'state_dict')
        }

# --- Convenience functions for direct use or testing ---

def predict_single_image(image_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick function to initialize the class and predict a single image."""
    detector = RobCropInference(model_path)
    return detector.predict(image_path)

def predict_from_camera(camera_index: int = 0, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Capture a single frame from a camera and predict."""
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {'error': f'Cannot open camera index {camera_index}'}
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # OpenCV captures in BGR, convert to RGB for PIL/PyTorch
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detector = RobCropInference(model_path)
            return detector.predict(rgb_frame)
        else:
            return {'error': 'Failed to capture image from camera'}
    except Exception as e:
        return {'error': str(e)}

def test_inference() -> bool:
    """A simple test to ensure the inference system runs without crashing."""
    print("🧪 Testing RobCrop Inference System")
    print("=" * 40)
    try:
        detector = RobCropInference()
        print("✅ Model initialized successfully")
        
        # Test with a dummy image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        result = detector.predict(dummy_image)
        
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
            return False
        else:
            print(f"✅ Prediction test passed:")
            print(f"   Disease: {result['disease']}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print(f"   Action: {result['action']}")
            return True
    except Exception as e:
        print(f"❌ Test failed during initialization: {str(e)}")
        return False

if __name__ == "__main__":
    test_inference()
