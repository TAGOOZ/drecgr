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
st.write("Upload an ECG beat image or use the camera to take a photo, crop it, and then the model will classify it as either **Normal** or **Arrhythmia**.")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an ECG image file", type=["png", "jpg", "jpeg"])

# Camera input for capturing an image
camera_image = st.camera_input("Capture image from camera")

# Initialize image to None
image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

elif camera_image is not None:
    # Convert the camera input image to PIL format
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_column_width=True)

# Option to crop image
if image is not None:
    st.write("Crop the image")
    left = st.slider("Left", 0, image.width, 0)
    upper = st.slider("Upper", 0, image.height, 0)
    right = st.slider("Right", 0, image.width, image.width)
    lower = st.slider("Lower", 0, image.height, image.height)
    if st.button("Crop Image"):
        image = image.crop((left, upper, right, lower))
        st.image(image, caption="Cropped Image", use_column_width=True)

# After cropping, submit for classification
if image is not None and st.button("Classify Image"):
    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

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

    # Provide medical advice based on the prediction
    if predicted_class == "arrhythmia":
        st.write(f"### You should visit a doctor")
    else:
        st.write(f"### بطل دلع يا راجل صحتك زي الحصان")
