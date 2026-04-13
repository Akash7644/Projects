
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ===============================
# Encoder (same as training)
# ===============================

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


# ===============================
# Classes (same order as training)
# ===============================

classes = [
    "ashok leaves",
    "banana leaves",
    "blackboard leaves",
    "gulmohar leaves",
    "jamun leaves",
    "lily leaves",
    "neem leaves",
    "paper flower leaves",
    "sadabahar(madagascar) leaves"
]

num_classes = len(classes)


# ===============================
# Load Model
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder()

classifier = nn.Sequential(
    nn.Linear(256,128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128,num_classes)
)

model = nn.Sequential(encoder, classifier)

model.load_state_dict(torch.load("leaf_classifier.pth", weights_only=True))

model = model.to(device)
model.eval()


# ===============================
# Image Transform (same as validation)
# ===============================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor()
])


# ===============================
# Prediction Function
# ===============================

def predict(image_path):

    image = Image.open(image_path).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():

        outputs = model(image)

        _, pred = torch.max(outputs,1)

    return classes[pred.item()]


# ===============================
# Test Prediction
# ===============================

image_path = r"C:\Users\akash\NN lab\mini_project\gul.jpg"
prediction = predict(image_path)

print("Predicted class:", prediction)

