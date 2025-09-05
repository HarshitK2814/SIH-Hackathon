"""
RobCrop FastAPI Backend
Agricultural Disease Detection API for SIH Hackathon
"""
import sys
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import io
import time
from PIL import Image
import logging
from typing import Optional, List

# --- FIX: Add current directory to path to ensure 'inference' can be found ---
# This makes the import robust, regardless of how the script is launched.
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
# ----------------------------------------------------------------------------
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
    allow_origins=["*"],  # In production, specify your frontend's exact origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded once at startup)
detector: Optional[RobCropInference] = None

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
        # This will stop the server from starting if the model fails to load
        raise

@app.get("/")
async def root():
    """API root endpoint providing basic service info."""
    return {
        "message": "🌱 RoboCrop Agricultural Disease Detection API",
        "status": "online",
        "version": "1.0.0",
        "model_accuracy": "96.45%"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint to verify service status."""
    return {
        "status": "healthy",
        "model_loaded": detector is not None,
        "timestamp": time.time(),
        "service": "RoboCrop Agricultural AI"
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """
    Predict crop disease from an uploaded image.
    """
    if detector is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. The service may be starting up or has failed."
        )
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        start_time = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result = detector.predict(image)
        if 'error' in result:
            raise Exception(result['error'])
            
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        treatment = get_treatment_advice(result['disease'])
        
        response = {
            "disease": result['disease'],
            "confidence": result['confidence'],
            "action": result['action'],
            "treatment": treatment,
            "processing_time_ms": round(processing_time, 2),
            "status": "success"
        }
        
        logger.info(f"Prediction: {result['disease']} ({result['confidence']:.1%})")
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Prediction error for file '{file.filename}': {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"An error occurred during prediction: {str(e)}"
        )

def get_treatment_advice(disease: str) -> str:
    """Get treatment recommendations for detected diseases."""
    treatments = {
        "Pepper,_bell___Bacterial_spot": "Apply copper-based bactericide. Remove infected leaves. Improve air circulation and avoid overhead watering.",
        "Pepper,_bell___healthy": "No treatment needed. Maintain good cultural practices and monitor regularly.",
        "Potato___Early_blight": "Apply fungicide (chlorothalonil or mancozeb). Remove infected foliage. Ensure adequate spacing.",
        "Potato___Late_blight": "Apply preventive fungicide immediately. Remove all infected plant material. Avoid overhead irrigation.",
        "Potato___healthy": "No treatment needed. Continue proper field management and disease monitoring.",
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Set to False in production
        log_level="info"
    )
