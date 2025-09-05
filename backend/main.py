"""
RobCrop FastAPI Backend
Agricultural Disease Detection API for SIH Hackathon
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import time
from PIL import Image
import logging

# Import your existing inference system
from inference import RobCropInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="🌱 RoboCrop Agricultural AI API",
    description="AI-powered crop disease detection for smart agriculture",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global detector
    try:
        logger.info("🚀 Initializing RobCrop AI model...")
        detector = RobCropInference()
        logger.info("✅ RobCrop AI model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "🌱 RoboCrop Agricultural Disease Detection API",
        "status": "online",
        "version": "1.0.0",
        "model_accuracy": "96.45%"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": time.time(),
        "service": "RobCrop Agricultural AI"
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict crop disease from uploaded image
    
    Returns:
        - disease: Detected disease name
        - confidence: Model confidence (0-1)
        - action: Recommended action (SPRAY/SKIP)
        - treatment: Treatment recommendation
        - processing_time: Inference time in milliseconds
    """
    
    # Validate model is loaded
    if detector is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please restart the service."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Read and process image
        start_time = time.time()
        contents = await file.read()
        
        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format: {str(e)}"
            )
        
        # Get prediction from your trained model
        result = detector.predict(image)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Get treatment recommendation
        treatment = get_treatment_advice(result['disease'])
        
        # Enhanced response
        response = {
            "disease": result['disease'],
            "confidence": result['confidence'],
            "action": result['action'],
            "treatment": treatment,
            "processing_time_ms": round(processing_time, 2),
            "class_index": result.get('class_index', -1),
            "status": "success"
        }
        
        logger.info(f"Prediction: {result['disease']} ({result['confidence']:.1%})")
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple images"""
    
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size limited to 10 images")
    
    try:
        results = []
        start_time = time.time()
        
        for i, file in enumerate(files):
            if not file.content_type or not file.content_type.startswith('image/'):
                continue
                
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            result = detector.predict(image)
            result['image_index'] = i
            result['filename'] = file.filename
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "total_images": len(results),
            "total_processing_time_ms": round(total_time, 2),
            "avg_time_per_image_ms": round(total_time / len(results), 2) if results else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

def get_treatment_advice(disease: str) -> str:
    """Get treatment recommendations for detected diseases"""
    
    treatments = {
        # Pepper diseases
        "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericide. Remove infected leaves. Improve air circulation and avoid overhead watering.",
        "Pepper,_bell___healthy": "No treatment needed. Maintain good cultural practices and monitor regularly.",
        
        # Potato diseases  
        "Potato___Early_blight": "Apply fungicide (chlorothalonil or mancozeb). Remove infected foliage. Ensure adequate spacing.",
        "Potato___Late_blight": "Apply preventive fungicide immediately. Remove all infected plant material. Avoid overhead irrigation.",
        "Potato___healthy": "No treatment needed. Continue proper field management and disease monitoring.",
        
        # Tomato diseases
        "Tomato___Bacterial_spot": "Use copper sprays weekly. Remove infected leaves immediately. Improve ventilation.",
        "Tomato___Early_blight": "Apply fungicide (boscalid or azoxystrobin). Mulch soil to prevent splash. Prune lower leaves.",
        "Tomato___Late_blight": "Emergency fungicide application required. Remove all infected plants immediately.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "No cure available. Remove infected plants. Control whitefly vectors.",
        "Tomato___Tomato_mosaic_virus": "No chemical treatment. Remove infected plants. Disinfect tools between plants.",
        "Tomato___healthy": "No treatment needed. Maintain optimal growing conditions and regular monitoring."
    }
    
    if 'healthy' in disease.lower():
        return "✅ Plant is healthy! Continue regular monitoring and good agricultural practices."
    
    specific_treatment = treatments.get(disease)
    if specific_treatment:
        return f"🚨 {specific_treatment}"
    else:
        return "⚠️ Disease detected. Consult agricultural extension officer for specific treatment recommendations."

@app.get("/model-info")
async def get_model_info():
    """Get model information and statistics"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_name": "RobCrop ResNet50",
        "accuracy": "96.45%",
        "num_classes": 11,
        "supported_crops": ["Pepper", "Potato", "Tomato"],
        "diseases_detected": [
            "Bacterial Spot", "Early Blight", "Late Blight", 
            "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy"
        ],
        "architecture": "ResNet50 Transfer Learning",
        "training_images": "60,786 agricultural images",
        "device": str(detector.device) if hasattr(detector, 'device') else "unknown"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
