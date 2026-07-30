import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load labels
# -----------------------------
with open("labels.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

print("Loaded classes:", classes)

# -----------------------------
# Load model
# -----------------------------
model = models.mobilenet_v3_small(weights=None)

model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)

model.load_state_dict(
    torch.load(
        "models/emotion_mobilenetv3.pth",
        map_location=device
    )
)

model.to(device)
model.eval()

# -----------------------------
# Image preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Face detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(rgb)

        tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            outputs = model(tensor)

            probs = F.softmax(outputs, dim=1)

            confidence, prediction = torch.max(probs, 1)

        label = classes[prediction.item()]
        conf = confidence.item() * 100

        # Draw rectangle
        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        # Draw text
        text = f"{label} {conf:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Recognition", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
