import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


# Paths
DATA_DIR = "dataset"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

MODEL_PATH = "models/emotion_mobilenetv3.pth"


# Training settings
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load datasets
train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


classes = train_dataset.classes

print("Classes:")
print(classes)


# Save labels
with open("labels.txt", "w") as f:
    for label in classes:
        f.write(label + "\n")


# Load MobileNetV3
model = models.mobilenet_v3_small(
    weights="DEFAULT"
)


# Replace classifier
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)


model = model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# Training loop
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for images, labels in tqdm(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()


        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()


    train_accuracy = 100 * correct / total

    print(
        f"Loss: {running_loss:.4f} "
        f"Accuracy: {train_accuracy:.2f}%"
    )


    # Validation
    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs,1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()


    val_accuracy = 100 * correct / total

    print(
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


# Save model

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\nTraining complete!")
print("Saved:", MODEL_PATH)
