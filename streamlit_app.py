import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Load the trained model
@st.cache_resource
def load_model():
    model = models.resnet50(weights="ResNet50_Weights.DEFAULT")  # Load pre-trained weights
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Ensure 2 output classes
    model.load_state_dict(torch.load("resnet50_model.pth", map_location=torch.device('cpu')))  # Path to your saved model
    model.eval()
    return model

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Class labels
class_names = ["arrhythmia", "normal"]

# Title and instructions
st.title("ECG Beat Classification")
st.write("Upload an ECG beat image, and the model will classify it as either **Normal** or **Arrhythmia**.")

# File uploader
uploaded_file = st.file_uploader("Choose an ECG image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Load the model
    model = load_model()
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)  # Get probabilities
        confidence, predicted = torch.max(probabilities, 1)

    # Display prediction and confidence
    predicted_class = class_names[predicted.item()]
    confidence_score = confidence.item() * 100
    

    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"### Confidence: **{confidence_score:.2f}%**")
    if class_names[predicted.item()] == "Arrhythmia":  # "Arrhythmia" needs quotes
        st.write(f"### You should visit a doctor")
    else:
        st.write(f"### بطل دلع يا راجل صحتك زي الحصان")
