# 🧠 Advanced Emotion Recognition System

A state-of-the-art facial emotion recognition system combining ResNet50V2 deep learning with Google Gemini AI for intelligent insights.

## ✨ Features

- **Advanced Deep Learning**: ResNet50V2 with transfer learning and fine-tuning
- **7 Emotion Classes**: Happy, Sad, Angry, Surprise, Fear, Disgust, Neutral
- **AI-Powered Insights**: Google Gemini 2.5 Flash provides contextual analysis
- **Grad-CAM Visualization**: See what the model focuses on
- **Modern Web Interface**: Clean, responsive UI with upload and webcam support
- **FastAPI Backend**: High-performance REST API
- **Confidence Validation**: Only shows results when model is confident

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Webcam (for live detection)
- Google Gemini API key (free from [Google AI Studio](https://aistudio.google.com/app/apikey))

### Installation

1. **Clone and navigate to the project**
   ```bash
   cd facial-emotion-classification
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements-new.txt
   ```

4. **Set up environment variables**
   ```bash
   cd backend
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

### Training the Model (Optional - if you need better accuracy)

If the current model is not performing well, retrain it with the improved script:

```bash
python train_resnet_improved.py
```

This will:
- Phase 1: Train custom classification head (20 epochs)
- Phase 2: Fine-tune top 30 ResNet layers (30 epochs)
- Save the best model as `resnet_emotion_model.h5`

**Training tips:**
- Ensure you have at least 200+ images per emotion class
- Images should be properly labeled in the `data/` folder
- Training may take 1-2 hours depending on your hardware
- Use GPU if available for faster training

### Running the Application

1. **Start the FastAPI backend**
   ```bash
   cd backend
   python main.py
   ```
   
   Backend will run on `http://localhost:8000`
   API docs available at `http://localhost:8000/docs`

2. **Start the frontend**
   
   Open `frontend/index.html` in your browser, or use a simple HTTP server:
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   
   Then visit `http://localhost:3000`

## 📁 Project Structure

```
facial-emotion-classification/
├── backend/
│   ├── main.py              # FastAPI server with Gemini integration
│   ├── .env.example         # Environment template
│   └── .env                 # Your API keys (create this)
├── frontend/
│   ├── index.html           # Main UI
│   ├── style.css            # Styling
│   └── script.js            # Frontend logic
├── data/                    # Training images
├── train_resnet_improved.py # Improved training script
├── resnet_emotion_model.h5  # Trained model
├── requirements-new.txt     # Python dependencies
└── README.md
```

## 🎯 API Endpoints

### Health Check
```
GET /health
```

### Predict Emotion
```
POST /predict
Content-Type: multipart/form-data

Parameters:
- file: Image file (jpg, png, jpeg)
- include_heatmap: boolean (default: true)
- include_gemini: boolean (default: true)

Response:
{
  "emotion": "Happy",
  "confidence": 95.3,
  "all_probabilities": {...},
  "is_confident": true,
  "gemini_insight": "AI analysis...",
  "heatmap_base64": "..."
}
```

## 🔧 Configuration

### Backend (.env)
```env
GEMINI_API_KEY=your_key_here
MODEL_PATH=../resnet_emotion_model.h5
IMG_SIZE=224
```

### Frontend (script.js)
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

## 🎨 Usage

### Upload Image
1. Click "Upload Image" tab
2. Drag & drop or click to select an image
3. Choose options (Heatmap, AI Insights)
4. Click "Analyze Emotion"

### Live Webcam
1. Click "Live Webcam" tab
2. Click "Start Webcam"
3. Click "Capture & Analyze" to analyze current frame

## 🧪 Model Performance

The improved training script includes:
- **Two-phase training**: Custom head → Fine-tuning
- **Advanced augmentation**: Rotation, shift, zoom, brightness
- **Regularization**: Dropout, BatchNormalization
- **Callbacks**: Early stopping, learning rate reduction
- **Better architecture**: Deeper classification head

Expected accuracy: 75-85% (depends on your dataset quality)

## 🔍 Troubleshooting

### "Model not found" error
- Ensure `resnet_emotion_model.h5` exists in the root directory
- Check `MODEL_PATH` in backend/.env

### "API not reachable" error
- Make sure backend is running on port 8000
- Check `API_BASE_URL` in frontend/script.js

### Low accuracy / Wrong predictions
- **Retrain the model** using `train_resnet_improved.py`
- Ensure training data is properly labeled
- Add more diverse training images
- Check if images have clear, visible faces

### Gemini insights not showing
- Verify `GEMINI_API_KEY` is set correctly in .env
- Check API quota at [Google AI Studio](https://aistudio.google.com/)
- Gemini only activates when confidence > 60%

### Webcam not working
- Grant camera permissions in browser
- Use HTTPS or localhost (required by browsers)
- Try different browser if issues persist

## 🚀 Improving Model Accuracy

If you're getting "Surprise" predictions for everything:

1. **Retrain with balanced data**
   - Check class distribution: `python -c "import pandas as pd; print(pd.read_csv('data/labels.csv')['emotion'].value_counts())"`
   - Ensure each emotion has 200+ images
   
2. **Use the improved training script**
   ```bash
   python train_resnet_improved.py
   ```
   
3. **Better preprocessing**
   - The new script includes face detection validation
   - Images are properly normalized
   - Advanced augmentation prevents overfitting

4. **Monitor training**
   - Watch for validation accuracy > training accuracy (overfitting)
   - Early stopping prevents over-training
   - Best model is automatically saved

## 📊 Technology Stack

- **Backend**: FastAPI, TensorFlow/Keras, OpenCV
- **AI Models**: ResNet50V2, Google Gemini 2.5 Flash
- **Frontend**: Vanilla JavaScript, HTML5, CSS3
- **ML Tools**: NumPy, Pandas, Scikit-learn

## 🤝 Contributing

1. Improve model architecture
2. Add more emotion classes
3. Enhance UI/UX
4. Add batch processing
5. Implement real-time webcam emotion tracking

## 📝 License

MIT License - feel free to use for your projects!

## 🙏 Acknowledgments

- ResNet50V2: Microsoft Research
- Google Gemini: Google AI
- Dataset: [Your dataset source]

## 📧 Support

For issues and questions:
1. Check troubleshooting section
2. Review API docs at `/docs`
3. Open GitHub issue

---

**Made with ❤️ using AI + Deep Learning**
