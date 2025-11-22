import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import cv2
import numpy as np
import matplotlib.cm as cm

# === SETTINGS ===
IMG_SIZE = 224
MODEL_PATH = 'resnet_emotion_model.h5'
# Alphabetical order because of flow_from_dataframe
CLASS_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# === LOAD MODEL ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === GRAD-CAM (VISUALIZATION) ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="post_relu"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
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

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(jet_heatmap * 255)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    return superimposed_img

# === UI ===
st.title("🧠 Hybrid ResNet Emotion Recognition")

option = st.selectbox("Choose Input", ("Upload Image", "Live Webcam"))

if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Analyze'):
            # Preprocess using ResNet's specific requirement
            img_resized = cv2.resize(image_rgb, (IMG_SIZE, IMG_SIZE))
            # Crucial: Convert to float and preprocess like ResNet expects
            img_array = img_resized.astype('float32')
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Predict
            preds = model.predict(img_array)
            confidence = np.max(preds) * 100
            label_idx = np.argmax(preds)
            label = CLASS_LABELS[label_idx]
            
            # === THE LOGIC CHECK (Like LLM Validation) ===
            # If confidence is low, the model admits it doesn't know.
            st.markdown("---")
            if confidence < 55.0:
                st.warning(f"⚠️ **Uncertain Analysis ({confidence:.2f}%)**")
                st.write("The model detected input data that does not match known facial features clearly.")
                st.info("Result: **Inconclusive / Neutral**")
            else:
                st.success(f"✅ **Confirmed Analysis ({confidence:.2f}%)**")
                st.header(f"Emotion: {label}")
                
                # Heatmap
                try:
                    st.write("Analyzing feature map activations...")
                    # For ResNet50V2, the last conv layer is usually 'post_relu'
                    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="post_relu")
                    overlay = overlay_heatmap(image_rgb, heatmap)
                    st.image(overlay, caption='Model Attention (ResNet Features)', use_column_width=True)
                except:
                    pass

elif option == "Live Webcam":
    run = st.checkbox('Start Webcam')
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    
    while run:
        _, frame = camera.read()
        if frame is None: break
        frame = cv2.flip(frame, 1)
        
        # No face detection box - we feed the WHOLE image to ResNet
        # and let ResNet decide if it sees an emotion or not.
        
        img_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        img_array = img_resized.astype('float32')
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        preds = model.predict(img_array, verbose=0)
        confidence = np.max(preds) * 100
        label = CLASS_LABELS[np.argmax(preds)]
        
        # Logic Check Display
        if confidence > 60:
            color = (0, 255, 0) # Green
            text = f"{label} ({int(confidence)}%)"
        else:
            color = (0, 0, 255) # Red
            text = "Waiting for clear face..."
            
        cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    camera.release()