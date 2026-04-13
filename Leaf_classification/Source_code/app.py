import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# =========================
# Page Config
# =========================

st.set_page_config(
    page_title="Leaf Classifier",
    layout="wide"
)


# =========================
# Encoder
# =========================

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128*28*28,256),
            nn.ReLU()
        )

    def forward(self,x):
        return self.features(x)


# =========================
# Classes
# =========================

classes = [
    "ashok leaves","banana leaves","blackboard leaves",
    "gulmohar leaves","jamun leaves","lily leaves",
    "neem leaves","paper flower leaves",
    "sadabahar(madagascar) leaves"
]


# =========================
# Load Model (Cached)
# =========================

@st.cache_resource
def load_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder()

    classifier = nn.Sequential(
        nn.Linear(256,128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128,len(classes))
    )

    model = nn.Sequential(encoder, classifier)

    model.load_state_dict(
        torch.load("leaf_classifier.pth", map_location=device, weights_only=True)
    )

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()


# =========================
# Transform
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


# =========================
# UI
# =========================

st.title("🌿 Leaf Species Classifier")
st.caption("Self-Supervised Learning (SimCLR) based model")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg"])


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # LEFT SIDE → IMAGE
    with col1:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", width=300)
        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT SIDE → RESULTS
    with col2:

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs,1)

        prediction = classes[pred.item()]
        confidence = confidence.item()*100

        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("Class Probabilities")

        prob_dict = {
            classes[i]: float(probs[0][i]*100)
            for i in range(len(classes))
        }

        st.bar_chart(prob_dict)
