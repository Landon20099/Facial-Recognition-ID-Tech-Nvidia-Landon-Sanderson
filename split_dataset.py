import os
import shutil
import random

# Paths
BASE_DIR = "dataset"
SOURCE_DIR = os.path.join(BASE_DIR, "6 Emotions for image classification")

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def create_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def split_dataset():
    # Create train/val/test folders
    for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        create_folder(folder)

    emotions = os.listdir(SOURCE_DIR)

    print("Found emotions:")
    print(emotions)

    for emotion in emotions:
        source_emotion_dir = os.path.join(SOURCE_DIR, emotion)

        if not os.path.isdir(source_emotion_dir):
            continue

        # Get images
        images = [
            img for img in os.listdir(source_emotion_dir)
            if img.lower().endswith(IMAGE_EXTENSIONS)
        ]

        random.shuffle(images)

        total = len(images)

        train_end = int(total * TRAIN_RATIO)
        val_end = train_end + int(total * VAL_RATIO)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        print(f"\n{emotion}")
        print(f"Total: {total}")
        print(f"Train: {len(train_images)}")
        print(f"Val: {len(val_images)}")
        print(f"Test: {len(test_images)}")

        # Make emotion folders
        for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(
                os.path.join(folder, emotion),
                exist_ok=True
            )

        # Copy files
        for img in train_images:
            shutil.copy(
                os.path.join(source_emotion_dir, img),
                os.path.join(TRAIN_DIR, emotion, img)
            )

        for img in val_images:
            shutil.copy(
                os.path.join(source_emotion_dir, img),
                os.path.join(VAL_DIR, emotion, img)
            )

        for img in test_images:
            shutil.copy(
                os.path.join(source_emotion_dir, img),
                os.path.join(TEST_DIR, emotion, img)
            )

    print("\nDataset split complete!")


if __name__ == "__main__":
    split_dataset()
