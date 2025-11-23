"""
Test script for the Emotion Recognition API
Usage: python test_api.py path/to/image.jpg
"""

import sys
import requests
import json
from pathlib import Path

API_URL = "http://localhost:8000"

def test_health():
    """Test if API is running"""
    try:
        response = requests.get(f"{API_URL}/health")
        data = response.json()
        print("✅ API Health Check:")
        print(f"   Status: {data['status']}")
        print(f"   Model Loaded: {data['model_loaded']}")
        print(f"   Gemini Configured: {data['gemini_configured']}")
        return True
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure backend is running on port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict(image_path, include_heatmap=True, include_gemini=True):
    """Test emotion prediction"""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print(f"\n🔍 Analyzing: {image_path}")
    print("-" * 50)
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            params = {
                'include_heatmap': include_heatmap,
                'include_gemini': include_gemini
            }
            
            response = requests.post(f"{API_URL}/predict", files=files, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"\n🎭 Detected Emotion: {data['emotion']}")
                print(f"📊 Confidence: {data['confidence']:.2f}%")
                print(f"✓ Is Confident: {data['is_confident']}")
                
                print(f"\n📈 All Probabilities:")
                sorted_probs = sorted(data['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
                for emotion, prob in sorted_probs:
                    bar = "█" * int(prob / 5)
                    print(f"   {emotion:10s} {prob:5.1f}% {bar}")
                
                if data.get('gemini_insight'):
                    print(f"\n🤖 Gemini Insight:")
                    print(f"   {data['gemini_insight']}")
                
                if data.get('heatmap_base64'):
                    print(f"\n🔥 Heatmap: Generated (base64 encoded)")
                
                print("\n" + "=" * 50)
                print("✅ Analysis Complete!")
                
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
    except Exception as e:
        print(f"❌ Error during prediction: {e}")

def main():
    print("=" * 50)
    print("🧠 Emotion Recognition API Test")
    print("=" * 50 + "\n")
    
    # Test health
    if not test_health():
        print("\n💡 Start the backend with: cd backend && python main.py")
        sys.exit(1)
    
    # Test prediction
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_predict(image_path)
    else:
        print("\n💡 Usage: python test_api.py path/to/image.jpg")
        print("\nTesting with sample images...")
        
        # Try to find sample images
        sample_images = ['happy.jpg', 'anger.jpg']
        for img in sample_images:
            if Path(img).exists():
                test_predict(img)
                break
        else:
            print("❌ No sample images found. Please provide an image path:")
            print("   python test_api.py your_image.jpg")

if __name__ == "__main__":
    main()
