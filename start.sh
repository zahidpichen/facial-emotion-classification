#!/bin/bash

echo "🚀 Starting Emotion Recognition System..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements-new.txt

# Check if .env exists
if [ ! -f "backend/.env" ]; then
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from template..."
    cp backend/.env.example backend/.env
    echo ""
    echo "⚠️  IMPORTANT: Please edit backend/.env and add your GEMINI_API_KEY"
    echo "Get your API key from: https://aistudio.google.com/app/apikey"
    echo ""
    read -p "Press Enter after you've added your API key..."
fi

# Check if model exists
if [ ! -f "resnet_emotion_model.h5" ]; then
    echo "⚠️  Model file not found!"
    echo "You need to train the model first:"
    echo "  python train_resnet_improved.py"
    echo ""
    read -p "Do you want to train the model now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python train_resnet_improved.py
    else
        echo "❌ Cannot start without model. Please train first."
        exit 1
    fi
fi

# Start backend in background
echo ""
echo "🚀 Starting FastAPI backend..."
cd backend
python main.py &
BACKEND_PID=$!
cd ..

# Wait a bit for backend to start
sleep 3

# Start frontend
echo ""
echo "🌐 Starting frontend server..."
cd frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ System is running!"
echo ""
echo "📍 Frontend: http://localhost:3000"
echo "📍 Backend API: http://localhost:8000"
echo "📍 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers..."

# Wait for Ctrl+C
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
