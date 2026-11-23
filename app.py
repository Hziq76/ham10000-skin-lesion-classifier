import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

st.set_page_config(page_title="HAM10000 Skin Lesion Classifier", layout="centered")
st.title("Dermatoscopic Skin Lesion Classification")
st.write("Upload a dermatoscopic image to predict the lesion category.")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis-like lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]


@st.cache_resource
def load_model():
  model = models.mobilenet_v2(weights=None)
  model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
  state_dict = torch.load("mobilenetv2_skin.pth", map_location="cpu")
  model.load_state_dict(state_dict)
  model.eval()
  return model


uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  st.image(image, caption="Uploaded Image", width="stretch")

  model = load_model()
  input_tensor = transform(image).unsqueeze(0)

  with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_idx = torch.argmax(probs).item()

  st.subheader(f"Predicted Class: {class_names[pred_idx]}")
  st.write(f"Confidence: {probs[pred_idx].item() * 100:.2f}%")