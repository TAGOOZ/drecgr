import streamlit as st
from PIL import Image
import torch
import numpy as np
from transformers import AutoModelForImageClassification, AutoImageProcessor
import cv2
import torch.nn.functional as F

# Load the model and processor
MODEL_PATH = "/workspaces/drecgr/resnet2"  # Update to point to your directory
model = AutoModelForImageClassification.from_pretrained(MODEL_PATH, use_safetensors=True)
processor = AutoImageProcessor.from_pretrained(MODEL_PATH)

# Class labels
labels = model.config.id2label

# Streamlit app title
st.title("ECG Classification App")
st.write("Upload an ECG image to get a diagnostic result!")

# File uploader
uploaded_file = st.file_uploader("Upload an ECG Image", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    # Convert image to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to a fixed size (e.g., 224x224)
    image = image.resize((224, 224))
    
    # Preprocess the image using the processor
    inputs = processor(images=image, return_tensors="pt")
    return inputs

def detect_ecg_signals(image):
    # Convert the image to grayscale for easier processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the image to detect the lines (ECG signals)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours which represent the ECG signal regions
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours from top to bottom (based on their vertical position)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])
    
    # Extract the bounding boxes of the contours (start and end of each signal)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    
    return bounding_boxes

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.image(image, caption="Uploaded ECG Image", use_column_width=True)
    st.write("Processing the image...")
    
    # Detect ECG signal regions
    bounding_boxes = detect_ecg_signals(image_np)
    
    # Process and classify each ECG signal individually
    for i, (x, y, w, h) in enumerate(bounding_boxes):
        # Crop the ECG signal region from the image
        cropped_image = image_np[y:y+h, x:x+w]
        cropped_image_pil = Image.fromarray(cropped_image)
        
        # Preprocess and predict
        inputs = preprocess_image(cropped_image_pil)
        outputs = model(**inputs)
        
        # Get predicted class (highest logit value)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
        
        # Apply softmax to get probabilities (confidence levels)
        probabilities = F.softmax(outputs.logits, dim=1)
        confidence = probabilities[0, predicted_class].item()  # Get confidence for the predicted class
        
        # Display the cropped signal, predicted class, and confidence
        st.image(cropped_image_pil, caption=f"ECG Signal {i + 1}", use_column_width=True)
        st.write(f"Predicted Class for Signal {i + 1}: {labels[predicted_class]}")
        st.write(f"Confidence: {confidence * 100:.2f}%")
