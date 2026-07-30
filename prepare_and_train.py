#!/usr/bin/env python3
"""
Prepare raw captured photos and train emotion classifier
Moves photos from raw_capture to train directory and trains the model
"""

import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

DATA_DIR = "dataset"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODEL_PATH = "models/emotion_mobilenetv3.pth"

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(description="Prepare data and train the emotion classifier")
    parser.add_argument("--resume", action="store_true", help="Resume training from the latest checkpoint")
    parser.add_argument(
        "--checkpoint",
        default="models/emotion_mobilenetv3_checkpoint.pt",
        help="Path to the training checkpoint to load/save"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the total number of epochs to train"
    )
    return parser.parse_args()


def save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path):
    """Save model, optimizer, and training state for resuming."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load a checkpoint and restore model and optimizer state."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return {
        "epoch": checkpoint["epoch"],
        "best_val_loss": checkpoint["best_val_loss"],
    }


def organize_raw_captures():
    """Move photos from raw_capture folders to train directory"""
    print("\n" + "="*60)
    print("📁 ORGANIZING CAPTURED PHOTOS")
    print("="*60)
    
    emotions = ["anger", "disgust", "fear", "happy", "pain", "sad"]
    total_moved = 0
    
    for emotion in emotions:
        raw_dir = Path(TRAIN_DIR) / emotion / "raw_capture"
        
        if raw_dir.exists():
            photos = list(raw_dir.glob("*.jpg"))
            
            if photos:
                print(f"\n😊 {emotion.upper()}: Moving {len(photos)} photos...")
                
                for photo in photos:
                    dest = raw_dir.parent / photo.name
                    shutil.move(str(photo), str(dest))
                    total_moved += 1
                
                # Remove empty raw_capture folder
                try:
                    raw_dir.rmdir()
                except:
                    pass
    
    print(f"\n✅ Organized {total_moved} photos total\n")
    return total_moved

def train_model(args):
    """Train emotion classifier on all photos"""
    print("="*60)
    print("🎓 TRAINING EMOTION CLASSIFIER")
    print("="*60)

    total_epochs = args.epochs if args.epochs is not None else EPOCHS
    checkpoint_path = Path(args.checkpoint)
    
    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load datasets
    print("\n📊 Loading training data...")
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )
    
    classes = train_dataset.classes
    print(f"✅ Classes: {classes}")
    print(f"✅ Training samples: {len(train_dataset)}")
    print(f"✅ Validation samples: {len(val_dataset)}")
    
    # Save labels
    with open("labels.txt", "w") as f:
        for label in classes:
            f.write(label + "\n")
    
    # Load model
    print("\n🤖 Loading MobileNetV3...")
    model = models.mobilenet_v3_small(weights="DEFAULT")
    
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes)
    )
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        resume_state = load_checkpoint(model, optimizer, checkpoint_path, device)
        if resume_state is not None:
            start_epoch = resume_state["epoch"]
            best_val_loss = resume_state["best_val_loss"]
            print(f"\n🔄 Resuming from epoch {start_epoch} using checkpoint {checkpoint_path}")
        else:
            print(f"\n⚠️  No checkpoint found at {checkpoint_path}; starting a new training run")
    
    # Training loop
    print("\n" + "="*60)
    print(f"▶️  STARTING TRAINING ({start_epoch}/{total_epochs} epochs completed)")
    print("="*60)
    
    for epoch in range(start_epoch, total_epochs):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.1f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]  ")
        with torch.no_grad():
            for images, labels in pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.1f}%'
                })
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
        
        # Save best model and checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  💾 Best model saved to {MODEL_PATH}\n")

        save_checkpoint(model, optimizer, epoch + 1, best_val_loss, checkpoint_path)
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"📁 Model saved: {MODEL_PATH}")
    print("="*60)

def main():
    args = parse_args()

    print("\n🎬 EMOTION CLASSIFIER - PREPARE & TRAIN")
    print("="*60)
    
    # Organize photos
    moved = organize_raw_captures()
    
    if moved == 0:
        print("⚠️  No raw photos found to organize")
        print("Using existing training data...\n")
    
    # Train model
    train_model(args)
    
    print("\n🎉 Done! Your model is ready to use.")
    print("   Run: python3 webcam_emotions.py")

if __name__ == "__main__":
    main()
