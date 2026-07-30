import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open("labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Create model
model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(classes))

# Load trained weights
model.load_state_dict(torch.load("models/emotion_mobilenetv3.pth", map_location=device))
model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load image
image = Image.open("test.jpg").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    outputs = model(image)
    probabilities = F.softmax(outputs, dim=1)

confidence, prediction = torch.max(probabilities, 1)

print(f"Prediction: {classes[prediction.item()]}")
print(f"Confidence: {confidence.item()*100:.2f}%")
