from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import cv2
import numpy as np
import google.generativeai as genai
from pydantic import BaseModel
from typing import Optional
import base64
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
IMG_SIZE = 224
MODEL_PATH = '../resnet_emotion_model.h5'
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Configure Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# === INITIALIZE FASTAPI ===
app = FastAPI(
    title="Emotion Recognition API",
    description="Advanced facial emotion recognition with AI insights",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === LOAD MODEL ===
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# === RESPONSE MODELS ===
class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    all_probabilities: dict
    is_confident: bool
    gemini_insight: Optional[str] = None
    heatmap_base64: Optional[str] = None

# === HELPER FUNCTIONS ===
def preprocess_image(image_bytes: bytes) -> tuple:
    """Preprocess image for model prediction"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize and preprocess
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = img_resized.astype('float32')
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    return img_array, img_rgb

def detect_face(image_rgb):
    """Detect face in image for better validation"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="post_relu"):
    """Generate Grad-CAM heatmap"""
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        logger.error(f"Grad-CAM error: {e}")
        return None

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on original image"""
    import matplotlib.cm as cm
    
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    return superimposed_img

async def get_gemini_insight(emotion: str, confidence: float, image_bytes: bytes) -> str:
    """Get AI-powered insight from Gemini 2.5 Flash"""
    if not GEMINI_API_KEY:
        return None
    
    try:
        # Initialize Gemini model
        model_gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Convert image to base64 for Gemini
        import io
        from PIL import Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(img_rgb)
        
        # Create prompt
        prompt = f"""You are an expert emotion analyst. Analyze this facial image and provide a brief, insightful analysis.

The ML model detected: {emotion} (confidence: {confidence:.1f}%)

Please provide:
1. Validation of whether the detected emotion seems accurate based on facial features
2. Key facial indicators you observe (eyes, mouth, eyebrows, etc.)
3. A brief psychological/contextual insight about this emotion
4. Any suggestions if the confidence is low

Keep the response concise (3-4 sentences), professional, and helpful."""

        # Generate response
        response = model_gemini.generate_content([prompt, pil_image])
        return response.text
    
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None

# === API ENDPOINTS ===
@app.get("/")
async def root():
    return {
        "message": "Emotion Recognition API",
        "version": "2.0.0",
        "status": "running",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gemini_configured": GEMINI_API_KEY is not None
    }

@app.post("/predict", response_model=EmotionResponse)
async def predict_emotion(
    file: UploadFile = File(...),
    include_heatmap: bool = True,
    include_gemini: bool = True
):
    """
    Predict emotion from uploaded image
    
    - **file**: Image file (jpg, png, jpeg)
    - **include_heatmap**: Generate Grad-CAM visualization
    - **include_gemini**: Get AI insight from Gemini
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use jpg/png")
    
    try:
        # Read image
        image_bytes = await file.read()
        
        # Preprocess
        img_array, img_rgb = preprocess_image(image_bytes)
        
        # Detect face (optional validation)
        has_face = detect_face(img_rgb)
        
        # Predict
        predictions = model.predict(img_array, verbose=0)[0]
        confidence = float(np.max(predictions) * 100)
        label_idx = int(np.argmax(predictions))
        emotion = CLASS_LABELS[label_idx]
        
        # Get all probabilities
        all_probs = {CLASS_LABELS[i]: float(predictions[i] * 100) for i in range(len(CLASS_LABELS))}
        
        # Determine if prediction is confident
        is_confident = confidence > 60.0 and has_face
        
        # Generate heatmap if requested
        heatmap_base64 = None
        if include_heatmap and is_confident:
            heatmap = make_gradcam_heatmap(img_array, model)
            if heatmap is not None:
                overlay = overlay_heatmap(img_rgb, heatmap)
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Get Gemini insight if requested
        gemini_insight = None
        if include_gemini and is_confident:
            gemini_insight = await get_gemini_insight(emotion, confidence, image_bytes)
        
        return EmotionResponse(
            emotion=emotion,
            confidence=confidence,
            all_probabilities=all_probs,
            is_confident=is_confident,
            gemini_insight=gemini_insight,
            heatmap_base64=heatmap_base64
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple images"""
    results = []
    
    for file in files:
        try:
            image_bytes = await file.read()
            img_array, _ = preprocess_image(image_bytes)
            
            predictions = model.predict(img_array, verbose=0)[0]
            confidence = float(np.max(predictions) * 100)
            label_idx = int(np.argmax(predictions))
            emotion = CLASS_LABELS[label_idx]
            
            results.append({
                "filename": file.filename,
                "emotion": emotion,
                "confidence": confidence,
                "is_confident": confidence > 60.0
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
